# %% tags=["remove_cell"]
from dependencies.substitutes.header import hw_agent

# %% [markdown] tags=["footer_1"]
# #### Update the device configuration file
# After measurement, we may store the measured device properties inside a new file to use in future experiments.
# The time-unique identifier ensures that it is easy to find back previously found measurement results.
# %% tags=["footer_2"]
hw_agent.quantum_device.to_json_file("./dependencies/configs", add_timestamp=True)
