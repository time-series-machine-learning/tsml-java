package tsml.data_containers;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.*;

public class TimeSeriesTest {
    
    private double first;
    private double second;
    private double third;
    private double fourth;
    private double[] array;
    private List<Double> list;
    private TimeSeries ts;
    
    @Before
    public void before() {
        first = 8.9;
        second = -2.4;
        third = Double.NaN;
        fourth = 6.3;
        array = new double[] {first, second, third, fourth};
        list = Arrays.stream(array).boxed().collect(Collectors.toList());
        ts = new TimeSeries(array);
    }
    
    @Test
    public void testCtorArray() {
        ts = new TimeSeries(array);
        assertArrayEquals(array, ts.toValueArray(), 0d);
    }

    @Test
    public void testCtorList() {
        ts = new TimeSeries(list);
        assertEquals(list, ts.getSeries());
    }
    
    @Test
    public void testCopyCtor() {
        assertEquals(ts, new TimeSeries(ts));
    }

    @Test
    public void testSize() {
        assertEquals(array.length, ts.getSeriesLength());
    }

    @Test
    public void testValidValueAt() {
        assertTrue(ts.hasValidValueAt(0));
        assertTrue(ts.hasValidValueAt(1));
        assertFalse(ts.hasValidValueAt(2));
        assertTrue(ts.hasValidValueAt(3));
    }

    @Test
    public void testGet() {
        for(int i = 0; i < ts.getSeriesLength(); i++) {
            assertEquals(new Double(array[i]), ts.get(i));
            assertEquals(array[i], ts.getValue(i), 0d);
            if(Double.isNaN(array[i])) {
                assertEquals(TimeSeries.DEFAULT_VALUE, ts.getOrDefault(i), 0d);
            }
        }
    }
    
    // todo test hslice
    // todo test vslice
    // todo test metadata / stats
    
}
