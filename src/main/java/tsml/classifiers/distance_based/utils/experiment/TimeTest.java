package tsml.classifiers.distance_based.utils.experiment;

import org.junit.Assert;
import org.junit.Test;

public class TimeTest {
    
    @Test
    public void testStrToNanos1m70s() {
        Assert.assertEquals(130000000000L, Time.strToNanos("1m70s"));
    }
    
    @Test
    public void testNanosToStr1m70s() {
        Assert.assertEquals("2m10s", Time.nanosToStr(130000000000L));
    }

    @Test
    public void testStrToNanos130s() {
        Assert.assertEquals(130000000000L, Time.strToNanos("130s"));
    }

    @Test
    public void testNanosToStr130s() {
        Assert.assertEquals("2m10s", Time.nanosToStr(130000000000L));
    }
    
    @Test
    public void testEquals() {
        Assert.assertEquals(Time.nanosToStr(130000000000L), Time.nanosToStr(130000000000L));
        Assert.assertNotEquals(Time.nanosToStr(130000000000L), Time.nanosToStr(130000000001L));
        Assert.assertEquals(Time.strToNanos("1h30m"), Time.strToNanos("90m"));
        Assert.assertEquals(Time.strToNanos("2m10s"), Time.strToNanos("130s"));
        Assert.assertEquals(Time.strToNanos("2m10s"), Time.strToNanos("1m70s"));
        Assert.assertEquals(Time.strToNanos("130s"), Time.strToNanos("1m70s"));
        Assert.assertNotEquals(Time.strToNanos("131s"), Time.strToNanos("1m70s"));
        Assert.assertEquals(Time.strToNanos("130s"), Time.strToNanos(Time.nanosToStr(Time.strToNanos("1m70s"))));
    }
    
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
