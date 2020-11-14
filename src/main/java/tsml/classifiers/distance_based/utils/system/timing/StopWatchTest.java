package tsml.classifiers.distance_based.utils.system.timing;

import java.util.concurrent.TimeUnit;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import static utilities.Utilities.sleep;

public class StopWatchTest {

    private StopWatch stopWatch;

    @Before public void before() {
        stopWatch = new StopWatch();
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
        Assert.assertEquals(stopWatch.elapsedTimeStopped(), 0);
        Assert.assertEquals(stopWatch.lapTime(), 0);
        Assert.assertEquals(stopWatch.lap(), 0);
        Assert.assertEquals(stopWatch.lapTimeStamp(), timeStamp, 100);
        Assert.assertEquals(stopWatch.timeStamp(), timeStamp, 100);
        Assert.assertEquals(stopWatch.timeStamp(), stopWatch.lapTimeStamp());
    }

    @Test
    public void testResetTime() {
        stopWatch.start();
        sleep(100);
        Assert.assertNotEquals(stopWatch.lap(), 0);
        stopWatch.stop();
        stopWatch.resetElapsed();
        Assert.assertEquals(stopWatch.elapsedTimeStopped(), 0);
        Assert.assertEquals(stopWatch.lap(), 0);
        Assert.assertEquals(stopWatch.lapTime(), 0);
    }

    @Test
    public void testResetClock() {
        stopWatch.start();
        long startTime = stopWatch.lap();
        stopWatch.resetClock();
        Assert.assertTrue(stopWatch.elapsedTime() > startTime);
        Assert.assertTrue(stopWatch.lapTime() >= startTime);
        Assert.assertEquals(stopWatch.lapTimeStamp(), stopWatch.timeStamp());
    }

    @Test
    public void testLap() throws InterruptedException {
        long sleepTime = TimeUnit.NANOSECONDS.convert(100, TimeUnit.MILLISECONDS);
        long tolerance = TimeUnit.NANOSECONDS.convert(500, TimeUnit.MILLISECONDS);
//        System.out.println("t: " + tolerance);
//        System.out.println("s: " + sleepTime);
        stopWatch.start();
        for(int i = 1; i <= 5; i++) {
            long sleep = TimeUnit.MILLISECONDS.convert(sleepTime, TimeUnit.NANOSECONDS);
            Thread.sleep(sleep);
            long elapsed = stopWatch.elapsedTime();
            long lap = stopWatch.lap();
            System.out.println(sleepTime + " " + lap + " " + elapsed);
            Assert.assertTrue(elapsed > sleepTime * i );
            Assert.assertTrue(elapsed < (sleepTime + tolerance) * i);
            Assert.assertTrue(lap > sleepTime );
            Assert.assertTrue(lap < (sleepTime + tolerance) );
        }
    }

    @Test
    public void testStop() {
        stopWatch.start();
        long startTime = stopWatch.elapsedTime();
        sleep(10);
        Assert.assertTrue(stopWatch.isStarted());
        stopWatch.stop();
        long stopTime = stopWatch.elapsedTimeStopped();
        Assert.assertTrue(stopTime > 0);
        Assert.assertFalse(stopWatch.isStarted());
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
        long time = stopWatch.elapsedTimeStopped();
        long addend = 10;
        stopWatch.add(addend);
        Assert.assertEquals(addend + time, stopWatch.elapsedTimeStopped());
        long prevTime = stopWatch.elapsedTimeStopped();
        stopWatch.add(stopWatch);
        Assert.assertEquals(prevTime * 2, stopWatch.elapsedTimeStopped());
    }

    @Test
    public void testSplit() {
        int delay = 100;
        long target = TimeUnit.NANOSECONDS.convert(delay, TimeUnit.MILLISECONDS);
        long tolerance = TimeUnit.NANOSECONDS.convert(10, TimeUnit.MILLISECONDS);
        stopWatch.start();
        try {
            Thread.sleep(delay);
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        long split1 = stopWatch.lap();
        long a = stopWatch.elapsedTime();
        Assert.assertEquals(a, split1, tolerance);
        try {
            Thread.sleep(delay);
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        long split2 = stopWatch.lap();
        Assert.assertEquals(stopWatch.elapsedTime(), split1 + split2, tolerance);
        try {
            Thread.sleep(delay);
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        long split3 = stopWatch.lap();
        Assert.assertEquals(stopWatch.elapsedTime(), split1 + split2 + split3, tolerance);
    }

    @Test()
    public void testLapWhenStopped() {
        stopWatch.start();
        sleep(100);
        Assert.assertEquals(stopWatch.lapAndStop(), stopWatch.lap());
    }
}
