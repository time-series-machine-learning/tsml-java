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
 
package tsml.classifiers.distance_based.utils.collections.params;

import com.beust.jcommander.internal.Lists;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.utils.strings.StrUtils;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_FLAG;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ParamSetTest {

    @Test
    public void testDuplicate() {
        final ParamSet paramSet = new ParamSet();
        paramSet.add("a", 5);
        try {
            paramSet.add("a", 6);
            Assert.fail("added duplicate parameter");
        } catch(Exception ignored) {}
    }
    
    @Test
    public void testSetAndGetOptions() {
        String aFlag = "a";
        int aValue = 1;
        ParamSet paramSet = new ParamSet(aFlag, aValue);
        String[] options = paramSet.getOptions();
        Assert.assertArrayEquals(options, new String[] {"-" + aFlag, String.valueOf(aValue)});
        ParamSet other = new ParamSet();
        try {
            other.setOptions(options);
        } catch(Exception e) {
            Assert.fail(e.getMessage());
        }
        Assert.assertNotNull(other.get(aFlag));
        final String o = other.get(aFlag);
        Assert.assertEquals(String.valueOf(o), String.valueOf(aValue));
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
        Assert.assertEquals(paramSet.toString(), "-a 1");
        Assert.assertFalse(paramSet.isEmpty());
        Assert.assertEquals(paramSet.size(), 1);
        Object value = paramSet.get(aFlag);
        Assert.assertEquals(value, aValue);
    }

    @Test
    public void testAddNameAndValueAndParamSet() {
        String aFlag = "a";
        LCSSDistance aValue = new LCSSDistance();
        String bFlag = WINDOW_FLAG;
        double bValue = 0.5;
        String cFlag = LCSSDistance.EPSILON_FLAG;
        double cValue = 0.2;
        ParamSet subParamSetB = new ParamSet(bFlag, bValue);
        ParamSet subParamSetC = new ParamSet(cFlag, cValue);
        ParamSet paramSet = new ParamSet(aFlag, aValue, Lists.newArrayList(subParamSetB, subParamSetC));
//        System.out.println(paramSet);
        aValue.setEpsilon(cValue);
        aValue.setWindow(bValue);
        Assert.assertEquals("-a \"tsml.classifiers.distance_based.distances.lcss.LCSSDistance "
            + "-w 0.5 -e 0.2\"", paramSet.toString());
        Assert.assertFalse(paramSet.isEmpty());
        Assert.assertEquals(paramSet.size(), 1);
        Object aValueOut = paramSet.get(aFlag);
        Assert.assertNotSame(aValueOut, aValue);
        Object list = ((ParamHandler) aValueOut).getParams().get(bFlag);
        Assert.assertEquals(list, bValue);
        list = ((ParamHandler) aValueOut).getParams().get(cFlag);
        Assert.assertEquals(list, cValue);

        try {
            Assert.assertNotSame(aValue, paramSet.get(aFlag));
        } catch(Exception e) {
            Assert.fail(e.toString());
        }

        try {
            // value may be in string form
            Assert.assertNotSame(aValue, new ParamSet(aFlag, StrUtils.toOptionValue(aValue)).get(aFlag, aValue));
        } catch(Exception e) {
            Assert.fail(e.toString());
        }

        try {
            Assert.assertEquals(cValue, subParamSetC.get(cFlag, cValue), -1d);
        } catch(Exception e) {
            Assert.fail(e.toString());
        }

        try {
            // value may be in string form
            Assert.assertEquals(cValue, new ParamSet().add(cFlag, String.valueOf(cValue)).get(cFlag, -1d), 0d);
        } catch(Exception e) {
            Assert.fail(e.toString());
        }
    }
    
    @Test
    public void testStrings() {
        String str = "string based parameter value";
        String value = StrUtils.toOptionValue(str);
        try {
            Object obj = StrUtils.fromOptionValue(value);
            Assert.assertEquals(str, obj);
        } catch(Exception e) {
            Assert.fail(e.toString());
        }
    }
    
}
