package tsml.classifiers.distance_based.utils.experiment;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class TimeSpanTest {
    
    @Test
    public void testTimeSpanConversion() {
        Assert.assertEquals(130000000000L, new TimeSpan(130000000000L).inNanos());
        Assert.assertEquals(-130000000000L, new TimeSpan(-130000000000L).inNanos());
        Assert.assertEquals(130000000000L, new TimeSpan("130s").inNanos());
        Assert.assertEquals(130000000000L, new TimeSpan("1m70s").inNanos());
        Assert.assertEquals(130000000000L, new TimeSpan("2m10s").inNanos());
        Assert.assertEquals(-130000000000L, new TimeSpan("-130s").inNanos());
        Assert.assertEquals(-130000000000L, new TimeSpan("-1m70s").inNanos());
        Assert.assertEquals(-130000000000L, new TimeSpan("-2m10s").inNanos());
        Assert.assertEquals("0:00:02:10", new TimeSpan("130s").asTimeStamp());
        Assert.assertEquals("-0:00:02:10", new TimeSpan("-130s").asTimeStamp());
    }
    
    @Test
    public void testSort() {
        final List<String> output =
                newArrayList("130s", "2m10s", "71s", "1m000001s").stream().map(TimeSpan::new).sorted()
                        .map(TimeSpan::toString).collect(Collectors.toList());
        final ArrayList<String> expected = newArrayList("1m1s", "1m11s", "2m10s", "2m10s");
        Assert.assertEquals(expected, output);
    }
    
    @Test
    public void testStrToNanos1m70s() {
        Assert.assertEquals(130000000000L, new TimeSpan("1m70s").inNanos());
    }
    
    @Test
    public void testNanosToStr1m70s() {
        Assert.assertEquals("2m10s", new TimeSpan(130000000000L).label());
    }

    @Test
    public void testStrToNanos130s() {
        Assert.assertEquals(130000000000L, new TimeSpan("130s").inNanos());
    }

    @Test
    public void testNanosToStr130s() {
        Assert.assertEquals("2m10s", new TimeSpan(130000000000L).label());
    }
    
    @Test
    public void testEquals() {
        Assert.assertEquals(new TimeSpan(130000000000L), new TimeSpan(130000000000L));
        Assert.assertNotEquals(new TimeSpan(130000000000L), new TimeSpan(130000000001L));
        Assert.assertEquals(new TimeSpan("1h30m"), new TimeSpan("90m"));
        Assert.assertEquals(new TimeSpan("2m10s"), new TimeSpan("130s"));
        Assert.assertEquals(new TimeSpan("2m10s"), new TimeSpan("1m70s"));
        Assert.assertEquals(new TimeSpan("130s"), new TimeSpan("1m70s"));
        Assert.assertNotEquals(new TimeSpan("131s"), new TimeSpan("1m70s"));
        Assert.assertEquals(new TimeSpan("130s"), new TimeSpan(new TimeSpan(new TimeSpan("1m70s"))));
    }
    
    @Test
    public void testNanosToStr1h() {
        Assert.assertEquals("1h", new TimeSpan(3600000000000L).label());
    }

    @Test
    public void testNanosToStr90m() {
        Assert.assertEquals("1h30m", new TimeSpan(5400000000000L).label());
    }

    @Test
    public void testNanosToStr26h87m102s() {
        Assert.assertEquals("1d3h28m42s", new TimeSpan(98922000000000L).label());
    }

    @Test
    public void testStrToNanos1h() {
        Assert.assertEquals(3600000000000L, new TimeSpan("1h").inNanos());
    }

    @Test
    public void testStrToNanos90m() {
        Assert.assertEquals(5400000000000L, new TimeSpan("90m").inNanos());
    }

    @Test
    public void testStrToNanos26h87m102s() {
        Assert.assertEquals(98922000000000L, new TimeSpan("26h87m102s").inNanos());
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStrToNanosInvalidUnit() {
        new TimeSpan("26z");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStrToNanosEmpty() {
        new TimeSpan("");
    }
}
