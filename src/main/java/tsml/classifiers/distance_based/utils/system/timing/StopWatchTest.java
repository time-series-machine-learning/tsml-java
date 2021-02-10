/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.utils.system.timing;

import java.util.concurrent.TimeUnit;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class StopWatchTest {

    private StopWatch stopWatch;

    @Before public void before() {
        stopWatch = new StopWatch();
    }

    @Test(expected = IllegalStateException.class)
    public void testGetStartTimeWhenStopped() {
        stopWatch.stop(false);
        Assert.assertFalse(stopWatch.isStarted());
        stopWatch.getStartTime();
    }

    @Test
    public void testGetStartTimeWhenStarted() {
        long timeStamp = System.nanoTime();
        stopWatch.start();
        Assert.assertTrue(stopWatch.isStarted());
        long startTime = stopWatch.getStartTime();
        Assert.assertTrue(startTime > timeStamp);
        Assert.assertTrue(startTime < timeStamp + TimeUnit.NANOSECONDS.convert(10, TimeUnit.MILLISECONDS));
    }

    @Test
    public void testReset() {
        stopWatch.start();
        stopWatch.stop();
        stopWatch.reset();
        Assert.assertEquals(stopWatch.getTime(), 0);
    }

    @Test
    public void testResetTime() {
        stopWatch.start();
        stopWatch.resetTime();
        Assert.assertEquals(stopWatch.getTime(), 0);
    }

    @Test
    public void testResetClock() {
        stopWatch.start();
        long startTime = stopWatch.getTime();
        stopWatch.resetTime();
        Assert.assertTrue(stopWatch.getStartTime() > startTime);
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
            long lapTime = stopWatch.lap();
//            System.out.println("l: " + lapTime);
            Assert.assertTrue(lapTime > sleepTime * i );
            Assert.assertTrue(lapTime < (sleepTime + tolerance) * i);
        }
    }

    @Test
    public void testStop() {
        stopWatch.start();
        long startTime = stopWatch.getStartTime();
        Assert.assertTrue(stopWatch.isStarted());
        stopWatch.stop();
        long stopTime = stopWatch.getTime();
        Assert.assertTrue(stopTime > 0);
        Assert.assertFalse(stopWatch.isStarted());
    }

    @Test
    public void testDoubleStop() {
        stopWatch.start();
        Assert.assertTrue(stopWatch.isStarted());
        stopWatch.stop();
        Assert.assertTrue(stopWatch.isStopped());
        stopWatch.stop(false);
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
        stopWatch.start(false);
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
        long time = stopWatch.getTime();
        long addend = 10;
        stopWatch.add(addend);
        Assert.assertEquals(addend + time, stopWatch.getTime());
        long prevTime = stopWatch.getTime();
        stopWatch.add(stopWatch);
        Assert.assertEquals(stopWatch.getTime(), prevTime * 2);
    }

    @Test(expected = IllegalStateException.class)
    public void testLapWhenStopped() {
        stopWatch.stop();
        stopWatch.lap();
    }
}
