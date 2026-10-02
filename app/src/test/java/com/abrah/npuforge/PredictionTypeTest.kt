package com.abrah.npuforge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PredictionTypeTest {
    @Test
    fun modelSpecValuesAreRecognised() {
        assertEquals(
            CheckpointInfo.PredictionType.V_PREDICTION,
            CheckpointInfo.PredictionType.fromMetadata("v"),
        )
        assertEquals(
            CheckpointInfo.PredictionType.EPSILON,
            CheckpointInfo.PredictionType.fromMetadata("epsilon"),
        )
    }

    @Test
    fun commonVPredictionSpellingIsRecognised() {
        assertEquals(
            CheckpointInfo.PredictionType.V_PREDICTION,
            CheckpointInfo.PredictionType.fromMetadata("v_prediction"),
        )
        assertEquals(
            CheckpointInfo.PredictionType.V_PREDICTION,
            CheckpointInfo.PredictionType.fromMetadata("V-Prediction"),
        )
    }

    @Test
    fun absentOrUnknownMetadataDoesNotGuess() {
        assertNull(CheckpointInfo.PredictionType.fromMetadata(null))
        assertNull(CheckpointInfo.PredictionType.fromMetadata("sample"))
    }

    @Test
    fun onlyVPredictionExportsTheCompatibilityMarker() {
        assertTrue(
            Converter.V_PRED_MARKER in
                Converter.predictionMarkers(CheckpointInfo.PredictionType.V_PREDICTION),
        )
        assertFalse(
            Converter.V_PRED_MARKER in
                Converter.predictionMarkers(CheckpointInfo.PredictionType.EPSILON),
        )
    }
}
