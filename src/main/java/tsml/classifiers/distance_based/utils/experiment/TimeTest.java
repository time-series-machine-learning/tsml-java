package tsml.classifiers.distance_based.utils.experiment;

import org.junit.Assert;
import org.junit.Test;

public class TimeTest {

    @Test
    public void testNanosToStr1h() {
        Assert.assertEquals("1h", Time.nanosToStr(3600000000000L));
    }

    @Test
    public void testNanosToStr90m() {
        Assert.assertEquals("1h30m", Time.nanosToStr(5400000000000L));
    }

    @Test
    public void testNanosToStr26h87m102s() {
        Assert.assertEquals("1d3h28m42s", Time.nanosToStr(98922000000000L));
    }

    @Test
    public void testStrToNanos1h() {
        Assert.assertEquals(3600000000000L, Time.strToNanos("1h"));
    }

    @Test
    public void testStrToNanos90m() {
        Assert.assertEquals(5400000000000L, Time.strToNanos("90m"));
    }

    @Test
    public void testStrToNanos26h87m102s() {
        Assert.assertEquals(98922000000000L, Time.strToNanos("26h87m102s"));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStrToNanosInvalidUnit() {
        Time.strToNanos("26z");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStrToNanosEmpty() {
        Time.strToNanos("");
    }
}
