package tsml.classifiers.distance_based.utils.strings;

import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.distances.transformed.BaseTransformDistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.transformers.Derivative;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Utils;

public class StrUtilsTest {

    @Test
    public void testSMOSetOptions() throws Exception {
        SMO smo = new SMO();
        ParamSet paramSet = new ParamSet();
        paramSet.add("-C", 5.5);
        ParamSet kernelParams = new ParamSet();
        kernelParams.add("-G", 0.5);
        paramSet.add("-K", new RBFKernel(), kernelParams);
        String[] options = paramSet.getOptions();
        smo.setOptions(options);
        Assert.assertEquals(5.5, smo.getC(), 0.0);
        Assert.assertTrue(smo.getKernel() instanceof RBFKernel);
        Assert.assertEquals(0.5, ((RBFKernel) smo.getKernel()).getGamma(), 0.0);
    }

    @Test
    public void testOptionsFormat() throws Exception {
        String[] strs = Utils.splitOptions("a \"b -c 5 -d 6\"");
        Assert.assertEquals("a", strs[0]);
        Assert.assertEquals("b -c 5 -d 6", strs[1]);
        Assert.assertEquals(2, strs.length);
        strs = Utils.splitOptions(strs[1]);
        Assert.assertEquals("b", strs[0]);
        Assert.assertEquals("-c", strs[1]);
        Assert.assertEquals("5", strs[2]);
        Assert.assertEquals("-d", strs[3]);
        Assert.assertEquals("6", strs[4]);
        Assert.assertEquals(5, strs.length);
    }

    @Test
    public void testToAndFromOptions() throws Exception {
        LCSSDistance lcss = new LCSSDistance();
        final TransformDistanceMeasure tdm = new BaseTransformDistanceMeasure("", new Derivative(), lcss);
        lcss.setEpsilon(6);
        lcss.setWindow(0.7);
        String[] strs = tdm.getOptions();
        Assert.assertEquals("-d", strs[2]);
        Assert.assertEquals("tsml.classifiers.distance_based.distances.lcss.LCSSDistance -w 0.7 -e 6.0", strs[3]);
        Assert.assertEquals("-t", strs[0]);
        Assert.assertEquals("tsml.transformers.Derivative", strs[1]);
        Assert.assertEquals(4, strs.length);
        String[] substrs = Utils.splitOptions(strs[3]);
        Assert.assertEquals("tsml.classifiers.distance_based.distances.lcss.LCSSDistance", substrs[0]);
        Assert.assertEquals("-e", substrs[3]);
        Assert.assertEquals("6.0", substrs[4]);
        Assert.assertEquals("-w", substrs[1]);
        Assert.assertEquals("0.7", substrs[2]);
        Assert.assertEquals(5, substrs.length);
        lcss.setEpsilon(-1);
        lcss.setWindow(-1);
        tdm.setOptions(strs);
        lcss = (LCSSDistance) tdm.getDistanceMeasure();
        Assert.assertEquals(6, lcss.getEpsilon(), 0.0d);
        Assert.assertEquals(0.7, lcss.getWindow(), 0d);
        final ParamSet paramSet = new ParamSet();
        paramSet.setOptions(strs);
        lcss.setEpsilon(-1);
        lcss.setWindow(1);
        tdm.setParams(paramSet);
        lcss = (LCSSDistance) tdm.getDistanceMeasure();
        Assert.assertEquals(6, lcss.getEpsilon(), 0.0d);
        Assert.assertEquals(0.7, lcss.getWindow(), 0d);
    }

    @Test
    public void testToOptionsValueNull() {
        Assert.assertEquals("null", StrUtils.toOptionValue(null));
    }

    @Test
    public void testToOptionsValueDouble() {
        Assert.assertEquals("5.786", StrUtils.toOptionValue(5.786));
    }

    @Test
    public void testToOptionsValueInt() {
        Assert.assertEquals("5", StrUtils.toOptionValue(5));
    }

    @Test
    public void testToOptionsValueString() {
        Assert.assertEquals("\"\\\"hello\\\"\"", StrUtils.toOptionValue("hello"));
    }

    @Test
    public void testToOptionsValueObject() {
        Assert.assertEquals("tsml.classifiers.distance_based.distances.lcss.LCSSDistance -w 1.0 -e 0.01", StrUtils.toOptionValue(new LCSSDistance()));
    }

    @Test
    public void testToOptionsValueStringWithWhiteSpace() throws Exception {
        Assert.assertEquals("\"\\\"hello goodbye\\\"\"", StrUtils.toOptionValue("hello goodbye"));
    }

    @Test
    public void testFromOptionsValueNull() throws Exception {
        Assert.assertNull(StrUtils.fromOptionValue("null", null));
    }

    @Test
    public void testFromOptionsValueDouble() throws Exception {
        Assert.assertEquals(new Double(5.786), StrUtils.fromOptionValue("5.786", Double::parseDouble));
    }

    @Test
    public void testFromOptionsValueDoubleNegative() throws Exception {
        Assert.assertEquals(new Double(-5.786), StrUtils.fromOptionValue("-5.786", Double::parseDouble));
    }

    @Test
    public void testFromOptionsValueDoubleNegativeMissingLeadingZero() throws Exception {
        Assert.assertEquals(new Double(-.786), StrUtils.fromOptionValue("-.786", Double::parseDouble));
    }

    @Test
    public void testFromOptionsValueDoubleMissingLeadingZero() throws Exception {
        Assert.assertEquals(new Double(.786), StrUtils.fromOptionValue(".786", Double::parseDouble));
    }

    @Test
    public void testFromOptionsValueInt() throws Exception {
        Assert.assertEquals(new Integer(5), StrUtils.fromOptionValue("5", Integer::parseInt));
    }

    @Test
    public void testFromOptionsValueBooleanTrue() throws Exception {
        Assert.assertEquals(Boolean.TRUE, StrUtils.fromOptionValue("true", Boolean::parseBoolean));
    }

    @Test
    public void testFromOptionsValueBooleanFalse() throws Exception {
        Assert.assertEquals(Boolean.FALSE, StrUtils.fromOptionValue("false", Boolean::parseBoolean));
    }

    @Test
    public void testFromOptionsValueString() throws Exception {
        Assert.assertEquals(StrUtils.fromOptionValue("\"\\\"hello\\\"\""), "hello");
    }

    @Test
    public void testFromOptionsValueStringWithWhiteSpace() throws Exception {
        Assert.assertEquals(StrUtils.fromOptionValue("\"\\\"hello goodbye\\\"\""), "hello goodbye");
    }

    @Test
    public void testFromOptionsValueObject() throws Exception {
        Object obj = StrUtils.fromOptionValue("tsml.classifiers.distance_based.distances.lcss.LCSSDistance -e 0.01 -ws -1");
        final LCSSDistance lcss = new LCSSDistance();
        // should be entirely different instances
        Assert.assertNotEquals(lcss, obj);
        // but be the same parameter-wise
        Assert.assertEquals(lcss.toString(), obj.toString());
    }

    @Test(expected = Exception.class)
    public void testFromOptionsValueEmptyString() throws Exception {
        Assert.assertEquals("", StrUtils.fromOptionValue(""));
    }

    @Test
    public void testIsFlagEmpty() throws Exception {
        Assert.assertFalse(StrUtils.isFlag(""));
    }

    @Test
    public void testIsFlagSingleLetter() throws Exception {
        Assert.assertTrue(StrUtils.isFlag("-b"));
    }

    @Test
    public void testIsFlagMultiLetter() throws Exception {
        Assert.assertTrue(StrUtils.isFlag("-bbbbbbb"));
    }

    @Test
    public void testIsFlagNoMinusSingleLetter() throws Exception {
        Assert.assertFalse(StrUtils.isFlag("b"));
    }

    @Test
    public void testIsFlagNoMinusMultiLetter() throws Exception {
        Assert.assertFalse(StrUtils.isFlag("bbbbbbb"));
    }

    @Test
    public void testIsOptionNoMinus() throws Exception {
        Assert.assertFalse(StrUtils.isOption("b", new String[] {"b", "c"}));
    }

    @Test
    public void testIsOptionFalse() throws Exception {
        Assert.assertFalse(StrUtils.isOption("-b", new String[] {"-b", "-c"}));
    }

    @Test
    public void testIsOptionTrue() throws Exception {
        Assert.assertTrue(StrUtils.isOption("-b", new String[] {"-b", "c"}));
    }

    @Test
    public void testIsOptionEmpty() throws Exception {
        Assert.assertFalse(StrUtils.isOption("", new String[] {"", "-c", "5"}));
    }
}
