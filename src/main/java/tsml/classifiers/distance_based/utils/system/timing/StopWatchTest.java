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
import tsml.classifiers.distance_based.utils.system.copy.CopierUtils;

import static utilities.Utilities.busyWait;

public class StopWatchTest {

    private StopWatch stopWatch;

    @Before public void before() {
        stopWatch = new StopWatch();
    }
    
    @Test
    public void testStartSpecificTime() {
        stopWatch.start(0);
        Assert.assertEquals(0, stopWatch.timeStamp());
        final long timeStamp = System.nanoTime();
        Assert.assertEquals(timeStamp, stopWatch.elapsedTime(), TimeUnit.NANOSECONDS.convert(10, TimeUnit.MILLISECONDS));
    }
    
    @Test
    public void testElapsedTime() {
        stopWatch.start();
        busyWait(delay);
        stopWatch.stop();
        final long l = stopWatch.elapsedTime();
        System.out.println(l);
        // should be at least delay of time elapsed
        Assert.assertTrue(l >= delay);
        // should be somewhere less than delay*2 elapsed. This varies from machine to machine, so the test is not very
        // stringent on tolerance
        Assert.assertTrue(l <= delay * 2);
    }
    
    @Test
    public void testSerialisation() {
        stopWatch.start();
        long timeStamp = stopWatch.timeStamp();
        StopWatch other = CopierUtils.deserialise(CopierUtils.serialise(stopWatch));
        Assert.assertTrue(other.isStopped());
        // make sure the clock / timeStamp gets reset post ser
        Assert.assertTrue(other.timeStamp() == 0);
        Assert.assertTrue(other.elapsedTime() > 0);


        stopWatch.resetAndStart();
        busyWait(delay);
        stopWatch.stop();
        timeStamp = stopWatch.timeStamp();
        other = CopierUtils.deserialise(CopierUtils.serialise(stopWatch));
        Assert.assertTrue(other.isStopped());
        // make sure the clock / timeStamp gets reset post ser
        Assert.assertEquals(other.timeStamp(), 0);
        Assert.assertEquals(other.elapsedTime(), stopWatch.elapsedTime());
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
        busyWait(delay);
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
    public void testStop() {
        stopWatch.start();
        long startTime = stopWatch.elapsedTime();
        busyWait(delay);
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
    
    private final long delay = TimeUnit.NANOSECONDS.convert(100, TimeUnit.MILLISECONDS);
}
