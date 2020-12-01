package tsml.data_containers;

import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static tsml.data_containers.TimeSeriesInstance.EMPTY_CLASS_LABELS;

public class TimeSeriesInstanceTest {
    
    private double[][] arrayA;
    private List<List<Double>> listA;
    private double[][] arrayB;
    private List<List<Double>> listB;
    private String[] classLabels;
    private String classLabelA;
    private String classLabelB;
    private int classLabelAIndex;
    private int classLabelBIndex;
    private TimeSeriesInstance instA;
    private TimeSeriesInstance instB;
    
    @Before
    public void before() {
        arrayA = new double[][] {
                {1,2,3,4},
                {5,6,7,8}
        };
        listA = Arrays.stream(arrayA).map(a -> Arrays.stream(a).boxed().collect(Collectors.toList())).collect(Collectors.toList());
        arrayB = new double[][] {
                {9,10,11,12},
                {13,14,15,16}
        };
        listB = Arrays.stream(arrayB).map(a -> Arrays.stream(a).boxed().collect(Collectors.toList())).collect(Collectors.toList());
        classLabelA = "A";
        classLabelB = "B";
        classLabels = new String[] {classLabelA, classLabelB};
        classLabelAIndex = 0;
        classLabelBIndex = 1;
        instA = new TimeSeriesInstance(arrayA, classLabelAIndex, classLabels);
        instB = new TimeSeriesInstance(arrayB, classLabelBIndex, classLabels);
    }
    
    @Test
    public void testCtorArray() {
        instB = new TimeSeriesInstance(arrayB);
        assertArrayEquals(EMPTY_CLASS_LABELS, instB.getClassLabels());
        assertEquals(-1, instB.getLabelIndex());
        assertEquals(null, instB.getClassLabel());
        assertEquals(Double.NaN, instB.getTargetValue(), 0d);
        assert2DArrayEquals(arrayB, instB.toValueArray());
    }
    
    @Test
    public void testCtorArrayLabelled() {
        instB = new TimeSeriesInstance(arrayB, classLabelBIndex, classLabels);
        assertArrayEquals(classLabels, instB.getClassLabels());
        assertEquals(classLabelBIndex, instB.getLabelIndex());
        assertEquals(classLabelB, instB.getClassLabel());
        assertEquals(classLabelBIndex, instB.getTargetValue(), 0d);
        assert2DArrayEquals(arrayB, instB.toValueArray());
    }
    
    @Test
    public void testToValueArrayA() {
        final double[][] array = instA.toValueArray();
        for(int i = 0; i < array.length; i++) {
            assertArrayEquals(arrayA[i], array[i], 0d);
        }
    }
    
    @Test
    public void testToValueArrayB() {
        final double[][] array = instB.toValueArray();
        for(int i = 0; i < array.length; i++) {
            assertArrayEquals(arrayB[i], array[i], 0d);
        }
    }
    
    @Test
    public void testGet() {
        for(int i = 0; i < arrayA.length; i++) {
            assertArrayEquals(arrayA[i], instA.get(i).toValueArray(), 0d);
            assertArrayEquals(arrayB[i], instB.get(i).toValueArray(), 0d);
        }
    }
    
    @Test
    public void testNumDimensions() {
        assertEquals(arrayA.length, instA.getNumDimensions());
        assertEquals(arrayB.length, instB.getNumDimensions());
    }

    @Test
    public void testCtorArrayRegressed() {
//        assertEquals(instA, new TimeSeriesInstance(array, 3d));
        // todo
    }
    
    @Test
    public void testCtorList() {
        instB = new TimeSeriesInstance(listB);
        assertArrayEquals(EMPTY_CLASS_LABELS, instB.getClassLabels());
        assertEquals(-1, instB.getLabelIndex());
        assertEquals(null, instB.getClassLabel());
        assertEquals(Double.NaN, instB.getTargetValue(), 0d);
        assert2DArrayEquals(arrayB, instB.toValueArray());
    }
    
    @Test
    public void testCtorListLabelled() {
        instB = new TimeSeriesInstance(listB, classLabelBIndex, classLabels);
        assertArrayEquals(classLabels, instB.getClassLabels());
        assertEquals(classLabelBIndex, instB.getLabelIndex());
        assertEquals(classLabelB, instB.getClassLabel());
        assertEquals(classLabelBIndex, instB.getTargetValue(), 0d);
        assert2DArrayEquals(arrayB, instB.toValueArray());
    }

    @Test
    public void testCtorListRegressed() {
//        instB = new TimeSeriesInstance(list, classLabelBIndex, classLabels);
//        assertEquals(instA, instB);
        // todo
    }
    
    @Test
    public void testClassLabel() {
        assertEquals(classLabelAIndex, instA.getLabelIndex());
        assertEquals(classLabelBIndex, instB.getLabelIndex());
        assertEquals(classLabelA, instA.getClassLabel());
        assertEquals(classLabelB, instB.getClassLabel());
        assertArrayEquals(classLabels, instA.getClassLabels());
        assertArrayEquals(classLabels, instB.getClassLabels());
    }
    
    // todo test target value (do in ctors?)
    // todo test hslice
    // todo test vslice
    // todo test metadata / stats
    
    public static void assert2DArrayEquals(double[][] expected, double[][] actual) {
        for(int i = 0; i < Math.max(expected.length, actual.length); i++) {
            assertArrayEquals(expected[i], actual[i], 0d);
        }
    }
}
