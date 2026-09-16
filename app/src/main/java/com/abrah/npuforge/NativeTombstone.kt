package com.abrah.npuforge

import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream

/** A bounded summary of Android's native crash protobuf; no protobuf runtime dependency.
 * Field numbers: https://android.googlesource.com/platform/system/core/+/refs/heads/main/debuggerd/proto/tombstone.proto
 * Unknown fields are skipped. Memory dumps, mapping lists and other threads are not printed.
 */
internal object NativeTombstone {
    private const val MAX_BYTES = 16 * 1024 * 1024
    private const val MAX_TEXT_BYTES = 4096
    private const val MAX_FRAMES = 24

    fun summarize(input: InputStream, expectedPid: Int): List<String> {
        val output = ByteArrayOutputStream()
        val buffer = ByteArray(8192)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            if (count == 0) throw IOException("native tombstone stream made no progress")
            if (output.size() > MAX_BYTES - count)
                throw IOException("native tombstone exceeds 16 MiB summary limit")
            output.write(buffer, 0, count)
        }
        val bytes = output.toByteArray()
        var pid = 0
        var tid = 0
        var abort = ""
        var signal = "unavailable"
        Reader(bytes).fields { field ->
            when (field.number) {
                5 -> pid = field.integer().toInt()
                6 -> tid = field.integer().toInt()
                10 -> {
                    var number = 0
                    var code = 0
                    var name = ""
                    var codeName = ""
                    field.message().fields { part ->
                        when (part.number) {
                            1 -> number = part.integer().toInt()
                            2 -> name = part.text()
                            3 -> code = part.integer().toInt()
                            4 -> codeName = part.text()
                        }
                    }
                    signal = "$name($number) code=$codeName($code)"
                }
                14 -> abort = field.text()
            }
        }
        if (pid != expectedPid)
            throw IOException("native tombstone PID $pid does not match exit PID $expectedPid")
        val lines = mutableListOf("Android native tombstone pid=$pid tid=$tid signal=$signal",
            "Android native abort message: ${abort.ifEmpty { "unavailable" }}")
        var foundThread = false
        // Protobuf fields and map entries may arrive in any order. Find the crash
        // TID before selecting its thread, and the map key before reading its value.
        Reader(bytes).fields { field ->
            if (field.number == 16 && !foundThread) {
                var key = 0
                var thread: Reader? = null
                field.message().fields { part ->
                    when (part.number) {
                        1 -> key = part.integer().toInt()
                        2 -> thread = part.message()
                    }
                }
                if (key == tid && thread != null) {
                    foundThread = true
                    var frames = 0
                    thread.fields { part ->
                        if (part.number == 4) {
                            if (frames < MAX_FRAMES) lines += frame(part.message(), frames)
                            frames++
                        }
                    }
                    if (frames == 0) lines += "Android native backtrace unavailable"
                    if (frames > MAX_FRAMES) lines += "Android native backtrace limited to $MAX_FRAMES frames"
                }
            }
        }
        if (!foundThread) lines += "Android native crashing-thread backtrace unavailable"
        return lines
    }

    private fun frame(reader: Reader, index: Int): String {
        var pc = 0L
        var function = ""
        var offset = 0L
        var file = ""
        var buildId = ""
        reader.fields { field ->
            when (field.number) {
                1 -> pc = field.integer()
                4 -> function = field.text()
                5 -> offset = field.integer()
                6 -> file = field.text()
                8 -> buildId = field.text()
            }
        }
        return "Android native #$index pc=0x${java.lang.Long.toUnsignedString(pc, 16)} " +
            "$file ($function+$offset) build_id=$buildId"
    }

    private class Field(
        val number: Int, private val wire: Int, private val value: Long,
        private val bytes: ByteArray, private val start: Int, private val end: Int,
    ) {
        fun integer(): Long {
            if (wire != 0) throw IOException("invalid integer wire type in native tombstone")
            return value
        }

        fun message(): Reader {
            if (wire != 2) throw IOException("invalid message wire type in native tombstone")
            return Reader(bytes, start, end)
        }

        fun text(): String {
            if (wire != 2) throw IOException("invalid string wire type in native tombstone")
            val size = minOf(end - start, MAX_TEXT_BYTES)
            return String(bytes, start, size, Charsets.UTF_8)
                .replace('\n', ' ').replace('\r', ' ').replace('\t', ' ') +
                if (size < end - start) " [truncated]" else ""
        }
    }

    private class Reader(
        private val bytes: ByteArray, private var position: Int = 0,
        private val end: Int = bytes.size,
    ) {
        fun fields(visit: (Field) -> Unit) {
            while (position < end) {
                val tag = varint()
                if (tag !in 1..0xffffffffL || tag ushr 3 == 0L)
                    throw IOException("invalid field tag in native tombstone")
                val wire = (tag and 7).toInt()
                val value = if (wire == 0) varint() else 0L
                val size = when (wire) {
                    0 -> 0L
                    1 -> 8L
                    2 -> varint()
                    5 -> 4L
                    else -> throw IOException("unsupported wire type in native tombstone")
                }
                if (size < 0 || size > end - position)
                    throw IOException("truncated field in native tombstone")
                val start = position
                position += size.toInt()
                visit(Field((tag ushr 3).toInt(), wire, value, bytes, start, position))
            }
        }

        private fun varint(): Long {
            var result = 0L
            for (index in 0..9) {
                if (position == end) throw IOException("truncated varint in native tombstone")
                val byte = bytes[position++].toInt() and 255
                if (index == 9 && byte > 1) throw IOException("overflowing varint in native tombstone")
                result = result or ((byte and 127).toLong() shl (index * 7))
                if (byte and 128 == 0) return result
            }
            throw IOException("invalid varint in native tombstone")
        }
    }
}
