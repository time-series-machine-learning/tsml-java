package tsml.classifiers.distance_based.utils.experiment;

import tsml.classifiers.distance_based.utils.strings.StrUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

import static java.util.concurrent.TimeUnit.*;

public class TimeSpan implements Comparable<TimeSpan>, Serializable {
    
    public TimeSpan(TimeSpan other) {
        this(other.inNanos());
    }
    
    public TimeSpan(long nanos) {
        this.nanos = nanos;
        if(nanos < 0) {
            nanos = Math.abs(nanos);
        }
        long remainingAmount = nanos;
        final TimeUnit[] values = TimeUnit.values();
        amounts = new long[TimeUnit.values().length];
        for(int i = values.length - 1; i >= 0; i--) {
            final TimeUnit unit = values[i];
            final long amount = unit.convert(remainingAmount, TimeUnit.NANOSECONDS);
            amounts[i] = amount;
            if(amount > 0) {
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
    }
    
    public TimeSpan(final String label) {
        this(labelToNanos(label));
    }
    
    private static long labelToNanos(String label) {
        if(label.isEmpty()) {
            throw new IllegalArgumentException("empty input");
        }
        boolean negative = false;
        int i = 0;
        if(label.charAt(0) == '-') {
            negative = true;
            i++;
        }
        StringBuilder sb = new StringBuilder();
        long totalNanos = 0;
        List<String> parts = new ArrayList<>();
        boolean previousIsDigit = true; // whether the last char was a digit or not
        for(; i < label.length(); i++) {
            final char c = label.charAt(i);
            final boolean isDigit = StrUtils.isDigit(c);
            final boolean isAlpha = StrUtils.isAlpha(c);
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
        for(int j = 0; j < parts.size(); j += 2) {
            final long value = Long.parseLong(parts.get(j));
            final String unit = parts.get(j + 1);
            totalNanos += TimeUnit.NANOSECONDS.convert(value, strToUnit(unit));
        }
        if(negative) {
            totalNanos = -totalNanos;
        }
        return totalNanos;
    }
    
    public static TimeUnit strToUnit(String str) {
        switch(str.toLowerCase()) {
            case "d": return DAYS;
            case "h": return HOURS;
            case "m": return MINUTES;
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

    private String label;
    private final long nanos;
    private final long[] amounts;

    /**
     * Get the specific amount of a given unit. I.e. take 1h33m50s. days() would give 0, hours() would give 1, minutes() would give 33 and seconds() would give 50.
     * @param unit
     * @return
     */
    public long get(TimeUnit unit) {
        return amounts[unit.ordinal()];
    }
    
    public long days() {
        return get(DAYS);
    }
    
    public long hours() {
        return get(HOURS);
    }

    public long minutes() {
        return get(MINUTES);
    }

    public long seconds() {
        return get(TimeUnit.SECONDS);
    }

    public long milliseconds() {
        return get(TimeUnit.MILLISECONDS);
    }

    public long microseconds() {
        return get(TimeUnit.MICROSECONDS);
    }

    public long nanoseconds() {
        return get(TimeUnit.NANOSECONDS);
    }

    /**
     * The time span expressed in nanoseconds.
     * @return
     */
    public long inNanos() {
        return nanos;
    }
    
    public String label() {
        if(label == null) {
            // the input label may be in an odd format, i.e. 90m. Therefore, convert the nanos back to a label which will output in maximised units, i.e. 1h30m. Use the nanos representation to do this.
            StringBuilder sb = new StringBuilder();
            if(nanos < 0) {
                sb.append("-");
            }
            for(int i = TimeUnit.values().length - 1; i >= 0; i--) {
                TimeUnit unit = TimeUnit.values()[i];
                long amount = get(unit);
                if(amount > 0) {
                    sb.append(amount).append(unitToStr(unit));
                }
            }
            label = sb.toString();
        }
        return label;
    }

    public String asTimeStamp() {
        StringBuilder sb = new StringBuilder();
        if(nanos < 0) {
            sb.append("-");
        }
        boolean first = true;
        for(int i = values().length - 1; i >= MINUTES.ordinal(); i--) {
            TimeUnit unit = values()[i];
            long amount = get(unit);
            if(!first) {
                sb.append(String.format("%02d", amount));
            } else if(amount > 0 || unit.equals(MINUTES)) {
                sb.append(amount).append(":");
                first = false;
            }
        }
        sb.append(String.format("%02d", seconds())).append(".").append(String.format("%03d", milliseconds()));
        return sb.toString();
    }

    @Override public String toString() {
        return asTimeStamp();
    }

    @Override public int compareTo(final TimeSpan other) {
        return Long.compare(inNanos(), other.inNanos());
    }

    @Override public boolean equals(final Object o) {
        if(this == o) {
            return true;
        }
        if(!(o instanceof TimeSpan)) {
            return false;
        }
        final TimeSpan timeSpan = (TimeSpan) o;
        return nanos == timeSpan.nanos;
    }

    @Override public int hashCode() {
        return Objects.hash(nanos);
    }
}
