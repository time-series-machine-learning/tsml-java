package tsml.classifiers.distance_based.utils.system.timing;

import java.util.concurrent.TimeUnit;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;

import static utilities.Utilities.sleep;

public class StopWatchTest {

    private StopWatch stopWatch;

    @Before public void before() {
        stopWatch = new StopWatch();
    }
    
    @Test
    public void testElapsedTime() {
        stopWatch.start();
        sleep(delay);
        stopWatch.stop();
        final long l = stopWatch.elapsedTime();
        Assert.assertEquals(TimeUnit.NANOSECONDS.convert(delay, TimeUnit.MILLISECONDS), l, tolerance);
    }

    @Test
    public void testLapTime() {
        stopWatch.start();
        long delay = 100;
        final long delayNanos = TimeUnit.NANOSECONDS.convert(delay, TimeUnit.MILLISECONDS);
        sleep(delay);
        Assert.assertEquals(delayNanos, stopWatch.elapsedTime(), tolerance);
        
        sleep(delay);
        Assert.assertEquals(2 * delayNanos, stopWatch.elapsedTime(), tolerance);
        
        sleep(delay);
        stopWatch.stop();
        Assert.assertEquals(delayNanos * 3, stopWatch.elapsedTime(), tolerance);
        Assert.assertTrue(stopWatch.isStopped());
        long timeStamp = System.nanoTime();
        
//        stopWatch.setStartTimeStamp(-1);
        Assert.assertEquals(delayNanos * 3, stopWatch.elapsedTime(), tolerance);

        timeStamp = System.nanoTime();
        stopWatch.reset();
        Assert.assertEquals(timeStamp, stopWatch.timeStamp(), 10000);
        Assert.assertEquals(0, stopWatch.elapsedTime());
        
    }
    
    @Test
    public void testSerialisation() {
        stopWatch.start();
        long timeStamp = stopWatch.timeStamp();
        StopWatch other = CopierUtils.deserialise(CopierUtils.serialise(stopWatch));
        Assert.assertTrue(other.isStopped());
        // make sure the clock / timeStamp gets reset post ser
        Assert.assertTrue(other.timeStamp() > timeStamp);
        Assert.assertTrue(other.elapsedTime() > 0);


        stopWatch.resetAndStart();
        sleep(delay);
        stopWatch.stop();
        timeStamp = stopWatch.timeStamp();
        other = CopierUtils.deserialise(CopierUtils.serialise(stopWatch));
        Assert.assertTrue(other.isStopped());
        // make sure the clock / timeStamp gets reset post ser
        Assert.assertEquals(other.timeStamp(), timeStamp, tolerance);
        Assert.assertEquals(other.elapsedTime(), stopWatch.elapsedTime());
    }

    @Test
    public void testGetPreviousElapsedTime() {
        final long time = System.nanoTime();
        stopWatch.start();
        int delay = 100;
        sleep(delay);
        long target = TimeUnit.NANOSECONDS.convert(delay, TimeUnit.MILLISECONDS);
        long tolerance = TimeUnit.NANOSECONDS.convert(10, TimeUnit.MILLISECONDS);
        Assert.assertEquals(target, stopWatch.elapsedTime(), tolerance);
    }
    
    @Test(expected = IllegalStateException.class)
    public void testGetStartTimeWhenStopped() {
        stopWatch.optionalStop();
        Assert.assertFalse(stopWatch.isStarted());
        stopWatch.stop();
    }

    @Test
    public void testGetStartTimeWhenStarted() {
        long timeStamp = System.nanoTime();
        stopWatch.start();
        Assert.assertTrue(stopWatch.isStarted());
        long startTime = stopWatch.timeStamp();
        Assert.assertTrue(startTime > timeStamp);
        Assert.assertTrue(startTime < timeStamp + TimeUnit.NANOSECONDS.convert(10, TimeUnit.MILLISECONDS));
    }

    @Test
    public void testReset() {
        stopWatch.start();
        stopWatch.stop();
        stopWatch.reset();
        long timeStamp = System.nanoTime();
        Assert.assertEquals(stopWatch.elapsedTime(), 0);
        Assert.assertEquals(stopWatch.timeStamp(), timeStamp, tolerance);
    }

    @Test
    public void testResetTime() {
        stopWatch.start();
        sleep(delay);
        Assert.assertNotEquals(stopWatch.elapsedTime(), 0);
        stopWatch.stop();
        stopWatch.resetElapsedTime();
        Assert.assertEquals(stopWatch.elapsedTime(), 0);
    }

    @Test
    public void testResetClock() {
        stopWatch.start();
        long startTime = stopWatch.timeStamp();
        stopWatch.reset();
        Assert.assertTrue(stopWatch.timeStamp() > startTime);
        Assert.assertEquals(stopWatch.timeStamp(), stopWatch.timeStamp());
    }
    
    @Test
    public void testLap() {
        long delay = TimeUnit.NANOSECONDS.convert(this.delay, TimeUnit.MILLISECONDS);
        stopWatch.start();
        for(int i = 1; i <= 5; i++) {
            long sleep = TimeUnit.MILLISECONDS.convert(delay, TimeUnit.NANOSECONDS);
            sleep(sleep);
            long elapsed = stopWatch.elapsedTime();
            Assert.assertTrue(elapsed > delay * i );
            Assert.assertTrue(elapsed < (delay + tolerance) * i);
        }
    }

    @Test
    public void testStop() {
        stopWatch.start();
        long startTime = stopWatch.elapsedTime();
        sleep(delay);
        Assert.assertTrue(stopWatch.isStarted());
        stopWatch.stop();
        long stopTime = stopWatch.elapsedTime();
        Assert.assertTrue(stopTime > 0);
        Assert.assertFalse(stopWatch.isStarted());
        Assert.assertTrue(stopWatch.isStopped());
        stopWatch.stopAndReset();
        Assert.assertFalse(stopWatch.isStarted());
        Assert.assertTrue(stopWatch.isStopped());
        Assert.assertEquals(0, stopWatch.elapsedTime());
    }

    @Test
    public void testDoubleStop() {
        stopWatch.start();
        Assert.assertTrue(stopWatch.isStarted());
        stopWatch.stop();
        Assert.assertTrue(stopWatch.isStopped());
        stopWatch.optionalStop();
        Assert.assertTrue(stopWatch.isStopped());
        try {
            stopWatch.stop();
            Assert.fail();
        } catch(IllegalStateException e) {

        }
        Assert.assertTrue(stopWatch.isStopped());
    }

    @Test
    public void testDoubleStart() {
        stopWatch.start();
        Assert.assertTrue(stopWatch.isStarted());
        stopWatch.optionalStart();
        Assert.assertTrue(stopWatch.isStarted());
        try {
            stopWatch.start();
            Assert.fail();
        } catch(IllegalStateException e) {

        }
        Assert.assertTrue(stopWatch.isStarted());
    }

    @Test
    public void testAdd() {
        stopWatch.start();
        stopWatch.stop();
        long time = stopWatch.elapsedTime();
        long addend = 10;
        stopWatch.add(addend);
        Assert.assertEquals(addend + time, stopWatch.elapsedTime());
        long prevTime = stopWatch.elapsedTime();
        stopWatch.add(stopWatch.elapsedTime());
        Assert.assertEquals(prevTime * 2, stopWatch.elapsedTime());
    }
    
    private final long tolerance = TimeUnit.NANOSECONDS.convert(10, TimeUnit.MILLISECONDS);
    
    private final long delay = 100;
    
    @Test
    public void testSplit() {
        long target = TimeUnit.NANOSECONDS.convert(delay, TimeUnit.MILLISECONDS);
        stopWatch.start();
        sleep(delay);
        Assert.assertEquals(target, stopWatch.elapsedTime(), tolerance);
        sleep(delay);
        Assert.assertEquals(stopWatch.elapsedTime(), target * 2, tolerance);
        sleep(delay);
        Assert.assertEquals(stopWatch.elapsedTime(), target * 3, tolerance);
    }
}
