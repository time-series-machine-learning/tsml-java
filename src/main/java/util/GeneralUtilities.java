package util;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import org.apache.commons.lang3.time.DurationFormatUtils;

public class GeneralUtilities {

	public static String getCurrentTimeStamp(String format) {
		return LocalDateTime.now().format(DateTimeFormatter.ofPattern(format));
	}
	
	public static String formatTime(long duration, String format) {
		return DurationFormatUtils.formatDuration((long) duration, "H:m:s.SSS");
	}
	
	public static void warmUpJavaRuntime() {
		//TODO
		System.out.println("TODO doing some extra work to warm up jvm..."
				+ "this helps to measure time more accurately for short experiments");
	}
	
	
}
