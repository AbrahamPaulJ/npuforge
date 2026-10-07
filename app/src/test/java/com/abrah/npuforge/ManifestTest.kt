package com.abrah.npuforge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * `swap_features.json` schema 2 ([Converter.manifestJson]). Plain JUnit: android.jar's
 * org.json is a stub here, so the checks are on the text -- which is also how the two
 * readers that matter read it.
 */
class ManifestTest {

    private fun sdxlSwap(kept: List<String>) = Converter.manifestJson(
        CheckpointInfo.Model.SDXL_SWAP, kept, "sdxl_swap_v2", 462,
        CheckpointInfo.PredictionType.EPSILON, "npuforge 1.0.12-preview", "SM8750",
    )

    /** Nightmare's backend (patch 020, `PipelineSdxl::readSwapInpaint`), verbatim in Kotlin. */
    private fun backendSeesInpaint(s: String): Boolean {
        val at = s.indexOf("\"features\"")
        return at >= 0 && s.indexOf("\"inp\"", at) >= 0
    }

    @Test
    fun featuresIsTheLastKeySoTheBackendCannotMisreadADetail() {
        val lora = sdxlSwap(listOf("lora", "cn", "ip"))
        assertTrue(lora.endsWith("\"features\":[\"lora\",\"cn\",\"ip\"]}"))
        assertFalse(backendSeesInpaint(lora))
        assertTrue(backendSeesInpaint(sdxlSwap(listOf("lora", "inp"))))
    }

    @Test
    fun detailCarriesTheTemplatesContractPerKeptFeature() {
        val m = sdxlSwap(listOf("lora", "cn", "ip"))
        assertTrue(m.contains("\"detail\":{\"lora\":{\"targets\":\"lora_targets.json\",\"rank\":64}," +
            "\"cn\":{\"residuals\":10,\"hint\":1024},\"ip\":{\"targets\":\"ip_targets.json\",\"layers\":70}}"))
        assertTrue(m.startsWith("{\"schema\":2,\"producer\":\"npuforge 1.0.12-preview\",\"template\":\"sdxl_swap_v2\","))
        assertTrue(m.contains("\"family\":\"sdxl\",\"kind\":\"swap\",\"prediction\":\"eps\",\"text_tokens\":462,\"size\":1024,\"soc\":\"SM8750\""))
    }

    @Test
    fun sd15SwapUsesItsOwnContract() {
        val m = Converter.manifestJson(
            CheckpointInfo.Model.SD15_SWAP, listOf("cn", "ip", "inp"), "swap_v3", 77,
            CheckpointInfo.PredictionType.V_PREDICTION, "npuforge x", "SM8750",
        )
        assertTrue(m.contains("\"cn\":{\"residuals\":13,\"hint\":512}"))
        assertTrue(m.contains("\"ip\":{\"targets\":\"ip_targets.json\",\"layers\":16}"))
        assertTrue(m.contains("\"inp\":{\"mask\":64}"))
        assertTrue(m.contains("\"prediction\":\"v\""))
    }

    @Test
    fun aPlainExportListsNoFeatures() {
        val m = Converter.manifestJson(
            CheckpointInfo.Model.SDXL, emptyList(), "template_sdxl", 231,
            CheckpointInfo.PredictionType.EPSILON, "npuforge x", "SM8750",
        )
        assertTrue(m.contains("\"kind\":\"plain\""))
        assertTrue(m.endsWith("\"detail\":{},\"features\":[]}"))
        assertFalse(backendSeesInpaint(m))
    }

    @Test
    fun featureOrderIsTheCanonicalOneWhateverTheSelectionOrder() {
        assertEquals(sdxlSwap(listOf("ip", "lora", "cn")), sdxlSwap(listOf("lora", "cn", "ip")))
    }
}
