# Nothing reflective here yet. Keep the entry points R8 cannot see.
-keep class com.abrah.npuforge.ConvertService { *; }
# The selected model family is carried by enum name in the service Intent.
-keep enum com.abrah.npuforge.CheckpointInfo$Model { *; }
