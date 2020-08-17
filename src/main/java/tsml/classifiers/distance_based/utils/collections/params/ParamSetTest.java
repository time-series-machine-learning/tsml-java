package tsml.classifiers.distance_based.utils.collections.params;

import com.beust.jcommander.internal.Lists;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.WarpingDistanceMeasure;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ParamSetTest {


    @Test
    public void testSetAndGetOptions() {
        String aFlag = "a";
        int aValue = 1;
        ParamSet paramSet = new ParamSet(aFlag, aValue);
        String[] options = paramSet.getOptions();
        Assert.assertArrayEquals(options, new String[] {"-" + aFlag, "\"" + String.valueOf(aValue) + "\""});
        ParamSet other = new ParamSet();
        try {
            other.setOptions(options);
        } catch(Exception e) {
            Assert.fail(e.getMessage());
        }
        Assert.assertNotNull(other.get(aFlag));
        Assert.assertEquals(other.get(aFlag).size(), 1);
        Assert.assertEquals(String.valueOf(other.get(aFlag).get(0)), String.valueOf(aValue));
    }

    @Test
    public void testEmptyToString() {
        ParamSet paramSet;
        paramSet = new ParamSet();
//        System.out.println(paramSet);
        Assert.assertEquals(paramSet.toString(), "");
    }

    @Test
    public void testHashcodeAndEquals() {
        String aFlag = "a";
        int aValue = 1;
        ParamSet paramSet = new ParamSet(aFlag, aValue);
        String bFlag = "a";
        int bValue = 1;
        ParamSet otherParamSet = new ParamSet(bFlag, bValue);
        String cFlag = "c";
        int cValue = 111;
        ParamSet unequalParamSet = new ParamSet(cFlag, cValue);
        Assert.assertNotEquals(paramSet, unequalParamSet);
        Assert.assertEquals(paramSet, otherParamSet);
        Assert.assertEquals(paramSet.hashCode(), otherParamSet.hashCode());
        Assert.assertNotEquals(otherParamSet, unequalParamSet);
    }

    @Test
    public void testAddNameAndValue() {
        String aFlag = "a";
        int aValue = 1;
        ParamSet paramSet = new ParamSet(aFlag, aValue);
//        System.out.println(paramSet);
        Assert.assertEquals(paramSet.toString(), "-a, \"1\"");
        Assert.assertFalse(paramSet.isEmpty());
        Assert.assertEquals(paramSet.size(), 1);
        List<Object> list = paramSet.get(aFlag);
        Assert.assertEquals(list.size(), 1);
        Assert.assertEquals(list.get(0), aValue);
    }

    @Test
    public void testAddNameAndMultipleValues() {
        String aFlag = "a";
        int aValue = 1;
        double anotherAValue = 3.3;
        String yetAnotherAValue = "not another!";
        ParamSet paramSet = new ParamSet(aFlag, aValue);
        paramSet.add(aFlag, anotherAValue);
        paramSet.add(aFlag, yetAnotherAValue);
//        System.out.println(paramSet);
        String out = "-a, \"1\", -a, \"3.3\", -a, \"\\\"not another!\\\"\"";
        Assert.assertEquals(paramSet.toString(), out);
        Assert.assertFalse(paramSet.isEmpty());
        Assert.assertEquals(paramSet.size(), 3);
        List<Object> list = paramSet.get(aFlag);
        Assert.assertEquals(list.size(), 3);
        Assert.assertEquals(list.get(0), aValue);
        Assert.assertEquals(list.get(1), anotherAValue);
        Assert.assertEquals(list.get(2), yetAnotherAValue);
    }

    @Test
    public void testAddNameAndValueAndParamSet() {
        String aFlag = "a";
        LCSSDistance aValue = new LCSSDistance();
        String bFlag = WarpingDistanceMeasure.WINDOW_SIZE_FLAG;
        int bValue = 5;
        String cFlag = LCSSDistance.EPSILON_FLAG;
        double cValue = 0.2;
        ParamSet subParamSetB = new ParamSet(bFlag, bValue);
        ParamSet subParamSetC = new ParamSet(cFlag, cValue);
        ParamSet paramSet = new ParamSet(aFlag, aValue, Lists.newArrayList(subParamSetB, subParamSetC));
//        System.out.println(paramSet);
        Assert.assertEquals(paramSet.toString(), "-a, \"tsml.classifiers.distance_based.distances.lcss.LCSSDistance "
            + "-e \"0.2\" -ws \"5\"\"");
        Assert.assertFalse(paramSet.isEmpty());
        Assert.assertEquals(paramSet.size(), 1);
        List<Object> list = paramSet.get(aFlag);
        Object aValueOut = list.get(0);
        Assert.assertEquals(list.size(), 1);
        Assert.assertEquals(aValueOut, aValue);
        list = ((ParamHandler) aValueOut).getParams().get(bFlag);
        Assert.assertEquals(list.size(), 1);
        Assert.assertEquals(list.get(0), bValue);
        list = ((ParamHandler) aValueOut).getParams().get(cFlag);
        Assert.assertEquals(list.size(), 1);
        Assert.assertEquals(list.get(0), cValue);
    }
}
