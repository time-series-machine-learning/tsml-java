package tsml.classifiers.distance_based.utils.experiment;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Time {
    
    public Time(final String inputLabel) {
        // convert the label to nanos
        nanos = strToNanos(inputLabel);
        // the input label may be in an odd format, i.e. 90m. Therefore, convert the nanos back to a label which will output in maximised units, i.e. 1h30m.
        label = nanosToStr(nanos);
    }
    
    public static String nanosToStr(long nanos) {
        final StringBuilder sb = new StringBuilder();
        long remainingAmount = nanos;
        final TimeUnit[] values = TimeUnit.values();
        for(int i = values.length - 1; i >= 0; i--) {
            final TimeUnit unit = values[i];
            final long amount = unit.convert(remainingAmount, TimeUnit.NANOSECONDS);
            if(amount > 0) {
                sb.append(amount).append(unitToStr(unit));
                // subtract the amount of time this accounts for in the remaining time. I.e. if the remaining time 
                // was 1.5 days (in nanos) then the amount would be only 1 day. We need to subtract the 1 day from 
                // the 1.5 days to give .5 days which can be represented in smaller units, i.e. hours.
                remainingAmount -= TimeUnit.NANOSECONDS.convert(amount, unit);
                // quit if remaining amount is 0
                if(remainingAmount == 0) {
                    break;
                }
            }
        }
        return sb.toString();
    }
    
    public static long strToNanos(String label) {
        if(label.isEmpty()) {
            throw new IllegalArgumentException("empty input");
        }
        StringBuilder sb = new StringBuilder();
        long totalNanos = 0;
        List<String> parts = new ArrayList<>();
        boolean previousIsDigit = true; // whether the last char was a digit or not
        for(int i = 0; i < label.length(); i++) {
            final char c = label.charAt(i);
            final boolean isDigit = isDigit(c);
            final boolean isAlpha = isAlpha(c);
            if(!isAlpha && !isDigit) {
                throw new IllegalArgumentException(c + " is not a digit or alpha character");
            }
            if(isAlpha && isDigit) {
                throw new IllegalStateException("the character cannot be a letter and a digit!! This should never happen!");
            }
            if(i != 0 && ((isDigit && !previousIsDigit) || (!isDigit && previousIsDigit))) {
                // was looking at value, now parsing unit
                // or vice versa
                // so store the contents of sb and reinitialise
                parts.add(sb.toString());
                sb = new StringBuilder();
            }
            sb.append(c);
            previousIsDigit = isDigit;
        }
        parts.add(sb.toString());
        if(parts.size() % 2 != 0) {
            throw new IllegalArgumentException("odd number of parts to string: " + label);
        }
        for(int i = 0; i < parts.size(); i += 2) {
            final long value = Long.parseLong(parts.get(i));
            final String unit = parts.get(i + 1);
            totalNanos += TimeUnit.NANOSECONDS.convert(value, strToUnit(unit));
        }
        return totalNanos;
    }
    
    public static boolean isAlpha(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }
    
    public static boolean isDigit(char c) {
        return (c >= '0' && c <= '9');
    }
    
    public static TimeUnit strToUnit(String str) {
        switch(str.toLowerCase()) {
            case "d": return TimeUnit.DAYS;
            case "h": return TimeUnit.HOURS;
            case "m": return TimeUnit.MINUTES;
            case "s": return TimeUnit.SECONDS;
            case "ms": return TimeUnit.MILLISECONDS;
            case "us": return TimeUnit.MICROSECONDS;
            case "ns": return TimeUnit.NANOSECONDS;
            default: throw new IllegalArgumentException("unknown unit conversion from: " + str);
        }
    }
    
    public static String unitToStr(TimeUnit unit) {
        switch(unit) {
            case DAYS: return "d";
            case HOURS: return "h";
            case MINUTES: return "m";
            case SECONDS: return "s";
            case MILLISECONDS: return "ms";
            case MICROSECONDS: return "us";
            case NANOSECONDS: return "ns";
            default: throw new IllegalArgumentException("unknown string label for unit: " + unit);
        }
    }

    private final String label;
    private final long nanos;

    public long inNanos() {
        return nanos;
    }
}
