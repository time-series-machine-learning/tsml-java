package tsml.classifiers.distance_based.proximity;

import java.util.concurrent.TimeUnit;
import org.junit.Assert;
import org.junit.Test;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class Duration {

    public Duration(double amount, TimeUnit unit) {
        setDuration(amount, unit);
    }

    private long nanos = 0;

    public void setDuration(double amount, TimeUnit unit) {
        boolean stop = false;
        long amountInteger;
        double amountRemainder;
        do {
            amountInteger = (long) amount;
            amountRemainder = amount - amountInteger;
            nanos += TimeUnit.NANOSECONDS.convert(amountInteger, unit);
            if(unit.equals(TimeUnit.NANOSECONDS)) {
                stop = true;
            } else {
                unit = TimeUnit.values()[unit.ordinal() - 1];
            }
        } while(!stop && amountRemainder > 0);
    }

    public double getAmount(TimeUnit unit) {
        double total = 0;
        boolean stop = false;
        long amountInteger;
        long amountRemainder = -1;
        long amount = nanos;
        do {
            // floors the nanos, therefore we need to grab the remainder
            amountInteger = unit.convert(amount, TimeUnit.NANOSECONDS);
            total += amountInteger;
            if(unit.equals(TimeUnit.NANOSECONDS)) {
                stop = true;
            } else {
                amountRemainder = amount - TimeUnit.NANOSECONDS.convert(amountInteger, unit);
            }
        } while(!stop && amountRemainder > 0);
        return total;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        boolean stop = false;
        long amountInteger;
        long amountRemainder = -1;
        long amount = nanos;
        TimeUnit unit = TimeUnit.DAYS;
        boolean first = true;
        do {
            // floors the nanos, therefore we need to grab the remainder
            amountInteger = unit.convert(amount, TimeUnit.NANOSECONDS);
            if(amountInteger > 0) {
                if(first) {
                    builder.append(amountInteger);
                } else {
                    long max = unit.convert(1, TimeUnit.values()[unit.ordinal() + 1]);
                    int pad = String.valueOf(max).length();
                }
                first = false;
            }
            if(unit.equals(TimeUnit.NANOSECONDS)) {
                stop = true;
            } else {
                amountRemainder = amount - TimeUnit.NANOSECONDS.convert(amountInteger, unit);
            }
        } while(!stop && amountRemainder > 0);
    }

    public static class Tests {
        @Test
        public void testSetDuration() {
            double amount = 2.55;
            TimeUnit unit = TimeUnit.HOURS;
            Duration duration = new Duration(amount, unit);
            System.out.println(duration);
            Assert.assertEquals(duration.toString(), "");
        }

        @Test
        public void testGetDuration() {
            double amount = 2.55;
            TimeUnit unit = TimeUnit.HOURS;
            Duration duration = new Duration(amount, unit);
            Assert.assertEquals(duration.getAmount(unit), amount, Double.MIN_VALUE);
        }
    }
}
