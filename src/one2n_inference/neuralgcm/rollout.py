"""Continuous NeuralGCM integration with persisted initialization forcing."""

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class NativeRollout:
    """Encode once and carry latent state between non-overlapping decoded chunks."""

    def __init__(self, model, initial_data, step_hours, seed):
        import jax

        self.model = model
        self.step_hours = step_hours
        inputs = model.inputs_from_xarray(initial_data.isel(time=0))
        initial_forcing = model.forcings_from_xarray(initial_data.isel(time=0))
        self.initial_forcing = initial_forcing
        self.state = model.encode(inputs, initial_forcing, jax.random.key(seed))
        self.forcing = model.forcings_from_xarray(initial_data.head(time=1))
        self.index = 0

    def _unroll(self, count):
        import jax

        self.state, predictions = self.model.unroll(
            self.state,
            self.forcing,
            steps=count,
            timedelta=np.timedelta64(self.step_hours, "h"),
            start_with_input=True,
        )
        self.index += count
        return jax.device_get(predictions)


class StreamingRollout(NativeRollout):
    """Decode P0 without advancing, then emit each single advanced native state."""

    def initialization(self):
        import jax

        decoded = self.model.decode(self.state, self.initial_forcing)
        decoded = jax.tree_util.tree_map(
            lambda value: np.asarray(value)[None], jax.device_get(decoded)
        )
        return self.model.data_to_xarray(decoded, times=np.array([self.index]))

    def advance(self):
        import jax

        self.state, predictions = self.model.unroll(
            self.state,
            self.forcing,
            steps=1,
            timedelta=np.timedelta64(self.step_hours, "h"),
            start_with_input=False,
        )
        self.index += 1
        return self.model.data_to_xarray(
            jax.device_get(predictions), times=np.array([self.index * self.step_hours])
        )


def run_model(model, initial_data, inner_steps, outer_steps, *, window, seed):
    """Return N native states including P0, retaining the original chunked route."""
    import jax

    rollout = NativeRollout(model, initial_data, inner_steps, seed)
    predictions = []
    for first in range(0, outer_steps, window):
        count = min(window, outer_steps - first)
        predictions.append(rollout._unroll(count))
        LOGGER.info("NeuralGCM: %d/%d states", first + count, outer_steps)
        jax.clear_caches()
    values = jax.tree_util.tree_map(lambda *arrays: np.concatenate(arrays, axis=0), *predictions)
    return model.data_to_xarray(values, times=np.arange(outer_steps) * inner_steps)
