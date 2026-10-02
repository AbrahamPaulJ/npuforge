# Epsilon and v-prediction checkpoints

Prediction type tells the sampler how to interpret the UNet output. It does not
change the UNet architecture or require a different QNN conversion. A converted
model can load and execute with the wrong type while producing broken images,
which is why the package must carry the setting to the runtime.

## Detection and override

NPuForge reads `modelspec.prediction_type` from the safetensors `__metadata__`
object. The [Stability AI ModelSpec 1.0.1 specification](https://github.com/Stability-AI/ModelSpec)
defines `v` and `epsilon`; NPuForge also recognises `v_prediction` and
`v-prediction`, which are common scheduler/config spellings.

The field is optional and community metadata is not always reliable. Therefore:

- recognised metadata selects the initial UI value;
- the user can override either value before conversion;
- absent or unrecognised metadata defaults to epsilon, matching every export
  before version 1.0.8;
- prediction type never follows the QAIRT/QNN version.

The conversion report records whether the selected value came from an explicit
choice, checkpoint metadata or the compatibility default.

## Export and runtime contract

For v-prediction, the exported ZIP contains an empty root entry named `V_PRED`.
For epsilon prediction, that entry is absent. LocalDream checks this marker and
passes `--use_v_pred` to its native backend. Nightmare Mobile 1.6.075 implements
the same package contract, so one converted ZIP works in both runtimes without
manual editing.

The marker affects scheduler-side interpretation only. The QNN graph, weight
conversion, calibration and compilation paths are identical for both prediction
types.
