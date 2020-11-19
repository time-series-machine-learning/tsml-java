package tsml.classifiers.distance_based.utils.experiment;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class TimeSpanTest {
    
    @Test
    public void testSort() {
        final List<String> output =
                newArrayList("none", "130s", "2m10s", "71s", "1m000001s", "none").stream().map(TimeSpan::new).sorted()
                        .map(TimeSpan::toString).collect(Collectors.toList());
        final ArrayList<String> expected = newArrayList("1m1s", "1m11s", "2m10s", "2m10s", "none", "none");
        Assert.assertEquals(expected, output);
    }
    
    @Test
    public void testNone() {
        final TimeSpan none = new TimeSpan("none");
        final TimeSpan some = new TimeSpan("70s");
        Assert.assertEquals(-1, none.inNanos());
        Assert.assertTrue(none.isNone());
        Assert.assertEquals(70000000000L, some.inNanos());
        Assert.assertFalse(some.isNone());
        Assert.assertEquals(0, none.compareTo(new TimeSpan("none")));
        Assert.assertTrue(none.compareTo(some) < 0);
        Assert.assertTrue(some.compareTo(none) > 0);
    }
    
    @Test
    public void testStrToNanos1m70s() {
        Assert.assertEquals(130000000000L, TimeSpan.strToNanos("1m70s"));
    }
    
    @Test
    public void testNanosToStr1m70s() {
        Assert.assertEquals("2m10s", TimeSpan.nanosToStr(130000000000L));
    }

    @Test
    public void testStrToNanos130s() {
        Assert.assertEquals(130000000000L, TimeSpan.strToNanos("130s"));
    }

    @Test
    public void testNanosToStr130s() {
        Assert.assertEquals("2m10s", TimeSpan.nanosToStr(130000000000L));
    }
    
    @Test
    public void testEquals() {
        Assert.assertEquals(TimeSpan.nanosToStr(130000000000L), TimeSpan.nanosToStr(130000000000L));
        Assert.assertNotEquals(TimeSpan.nanosToStr(130000000000L), TimeSpan.nanosToStr(130000000001L));
        Assert.assertEquals(TimeSpan.strToNanos("1h30m"), TimeSpan.strToNanos("90m"));
        Assert.assertEquals(TimeSpan.strToNanos("2m10s"), TimeSpan.strToNanos("130s"));
        Assert.assertEquals(TimeSpan.strToNanos("2m10s"), TimeSpan.strToNanos("1m70s"));
        Assert.assertEquals(TimeSpan.strToNanos("130s"), TimeSpan.strToNanos("1m70s"));
        Assert.assertNotEquals(TimeSpan.strToNanos("131s"), TimeSpan.strToNanos("1m70s"));
        Assert.assertEquals(TimeSpan.strToNanos("130s"), TimeSpan.strToNanos(TimeSpan.nanosToStr(TimeSpan.strToNanos("1m70s"))));
    }
    
    @Test
    public void testNanosToStr1h() {
        Assert.assertEquals("1h", TimeSpan.nanosToStr(3600000000000L));
    }

    @Test
    public void testNanosToStr90m() {
        Assert.assertEquals("1h30m", TimeSpan.nanosToStr(5400000000000L));
    }

    @Test
    public void testNanosToStr26h87m102s() {
        Assert.assertEquals("1d3h28m42s", TimeSpan.nanosToStr(98922000000000L));
    }

    @Test
    public void testStrToNanos1h() {
        Assert.assertEquals(3600000000000L, TimeSpan.strToNanos("1h"));
    }

    @Test
    public void testStrToNanos90m() {
        Assert.assertEquals(5400000000000L, TimeSpan.strToNanos("90m"));
    }

    @Test
    public void testStrToNanos26h87m102s() {
        Assert.assertEquals(98922000000000L, TimeSpan.strToNanos("26h87m102s"));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStrToNanosInvalidUnit() {
        TimeSpan.strToNanos("26z");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStrToNanosEmpty() {
        TimeSpan.strToNanos("");
    }
}
