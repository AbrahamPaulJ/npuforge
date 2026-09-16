package com.abrah.npuforge

import java.io.ByteArrayInputStream
import java.io.IOException
import java.io.InputStream

// Standalone fixtures: compile with NativeTombstone.kt, then run this main.
private fun varint(value: Long): ByteArray {
    var remaining = value
    val bytes = mutableListOf<Byte>()
    do {
        val last = remaining ushr 7 == 0L
        bytes += ((remaining and 127).toInt() or if (last) 0 else 128).toByte()
        remaining = remaining ushr 7
    } while (!last)
    return bytes.toByteArray()
}

private fun number(field: Int, value: Long) = varint((field * 8).toLong()) + varint(value)
private fun message(field: Int, value: ByteArray) =
    varint((field * 8 + 2).toLong()) + varint(value.size.toLong()) + value
private fun string(field: Int, value: String) = message(field, value.toByteArray())
private fun hex(value: String) = value.split(' ').map { it.toInt(16).toByte() }.toByteArray()

private fun rejects(bytes: ByteArray, expected: String) {
    try {
        NativeTombstone.summarize(ByteArrayInputStream(bytes), 42)
        error("accepted invalid fixture: $expected")
    } catch (e: IOException) {
        check(e.message.orEmpty().contains(expected)) { e.message.orEmpty() }
    }
}

fun main() {
    // Literal wire fixture from the AOSP schema: pid=42, tid=7, abort_message="OOM".
    val literal = hex("28 2a 30 07 72 03 4f 4f 4d")
    val basic = NativeTombstone.summarize(ByteArrayInputStream(literal), 42)
    check(basic[0] == "Android native tombstone pid=42 tid=7 signal=unavailable")
    check(basic[1] == "Android native abort message: OOM")
    check(basic.last().contains("backtrace unavailable"))

    val frame = number(1, 0x1234) + string(4, "Scudo::allocate") + number(5, 16) +
        string(6, "/system/lib64/libc.so") + string(8, "aabbcc")
    val otherThread = number(1, 8) + message(2, message(4, string(4, "wrong thread")))
    // The map value precedes its key, and the thread map precedes the top-level TID.
    val crashThread = message(2, message(4, frame)) + number(1, 7)
    val signal = number(1, 6) + string(2, "SIGABRT") + number(3, -1) + string(4, "SI_QUEUE")
    val unknown = number(1000, 88) + message(1001, byteArrayOf(0xff.toByte())) +
        varint(1002L * 8 + 1) + ByteArray(8) + varint(1003L * 8 + 5) + ByteArray(4)
    val reordered = message(16, otherThread) + message(16, crashThread) + unknown +
        message(10, signal) + literal
    val summary = NativeTombstone.summarize(ByteArrayInputStream(reordered), 42)
    check(summary.first().endsWith("signal=SIGABRT(6) code=SI_QUEUE(-1)"))
    check(summary.last() == "Android native #0 pc=0x1234 /system/lib64/libc.so (Scudo::allocate+16) build_id=aabbcc")
    check(summary.none { "wrong thread" in it })

    val manyFrames = (0 until 30).fold(ByteArray(0)) { bytes, _ -> bytes + message(4, frame) }
    val bounded = literal + message(16, number(1, 7) + message(2, manyFrames)) +
        string(14, "x".repeat(5000) + "\nspoof")
    val limited = NativeTombstone.summarize(ByteArrayInputStream(bounded), 42)
    check(limited.count { "Android native #" in it } == 24)
    check(limited.last().contains("limited to 24 frames"))
    check(limited[1].endsWith("[truncated]") && limited[1].length < 4200)
    val multiline = NativeTombstone.summarize(ByteArrayInputStream(literal + string(14, "a\nb\rc\td")), 42)
    check(multiline[1] == "Android native abort message: a b c d")

    rejects(byteArrayOf(), "PID 0")
    rejects(number(5, 99), "does not match")
    rejects(literal + hex("72 05 01"), "truncated field")
    rejects(literal + hex("80"), "truncated varint")
    rejects(literal + ByteArray(10) { 0xff.toByte() }, "overflowing varint")
    rejects(literal + hex("00"), "invalid field tag")
    rejects(literal + hex("73"), "unsupported wire type")
    rejects(literal + hex("70 01"), "invalid string wire type")
    rejects(literal + hex("72 ff ff ff ff ff ff ff ff ff 01"), "truncated field")

    val oversized = object : InputStream() {
        var readBytes = 0
        override fun read(): Int { readBytes++; return 0 }
        override fun read(bytes: ByteArray, off: Int, len: Int): Int {
            bytes.fill(0, off, off + len)
            readBytes += len
            return len
        }
    }
    try {
        NativeTombstone.summarize(oversized, 42)
        error("accepted oversized stream")
    } catch (e: IOException) {
        check(e.message.orEmpty().contains("16 MiB"))
        check(oversized.readBytes <= 16 * 1024 * 1024 + 8192)
    }
    println("Native tombstone fixtures passed: schema fields, reordered map, signed signal, unknown fields, bounds, malformed input")
}
