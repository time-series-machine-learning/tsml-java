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

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.continuous.ContinuousParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete.DiscreteParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.distribution.double_based.UniformDoubleDistribution;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_FLAG;
import static tsml.classifiers.distance_based.distances.dtw.spaces.DDTWDistanceSpace.newDDTWDistance;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ParamSpaceTest {

    public ParamSpaceTest() {
        // for users outside of this class, make sure we're all setup for tests
        before();
    }

    private int seed;
    private ParamSpace wParams;
    private List<Integer> wParamValues;
    private ParamSpace lParams;
    private UniformDoubleDistribution eDist;
    private UniformDoubleDistribution dDist;
    private DiscreteParamDimension<DistanceMeasure> wDmParams;
    private DiscreteParamDimension<DistanceMeasure> lDmParams;
    private ParamSpace params;

    @Before
    public void before() {
        seed = 0;
        wParamValues = buildWParamValues();
        wParams = buildWParams();
        lParams = buildLParams();
        eDist = buildEDist();
        dDist = buildDDist();
        wDmParams = buildWDmParams();
        lDmParams = buildLDmParams();
        params = buildParams();
    }

    public Random buildRandom() {
        return new Random(seed);
    }

    public List<Integer> buildWParamValues() {
        return Arrays.asList(1, 2, 3, 4, 5);
    }

    public ParamSpace buildWParams() {
        ParamMap wParams = new ParamMap();
        List<Integer> wParamValues = buildWParamValues();
        wParams.add(WINDOW_FLAG, wParamValues);
        return new ParamSpace(wParams);
    }

    public List<Integer> buildDummyValuesA() {
        return Arrays.asList(1,2,3,4,5);
    }

    public List<Integer> buildDummyValuesB() {
        return Arrays.asList(6,7,8,9,10);
    }

    public ParamSpace build2DDiscreteSpace() {
        ParamMap space = new ParamMap();
        List<Integer> a = buildDummyValuesA();
        List<Integer> b = buildDummyValuesB();
        space.add("a", a);
        space.add("b", b);
        return new ParamSpace(space);
    }

    public UniformDoubleDistribution buildDummyDistributionA() {
        UniformDoubleDistribution u = new UniformDoubleDistribution(0d, 0.5d);
        return u;
    }

    public UniformDoubleDistribution buildDummyDistributionB() {
        UniformDoubleDistribution u = new UniformDoubleDistribution(0.5d, 1d);
        return u;
    }

    public ParamSpace build2DContinuousSpace() {
        ParamMap space = new ParamMap();
        UniformDoubleDistribution a = buildDummyDistributionA();
        UniformDoubleDistribution b = buildDummyDistributionB();
        space.add("a", a);
        space.add("b", b);
        return new ParamSpace(space);
    }

    public UniformDoubleDistribution buildEDist() {
        UniformDoubleDistribution eDist = new UniformDoubleDistribution();
        eDist.setStart(0d);
        eDist.setEnd(0.25);
        return eDist;
    }

    public UniformDoubleDistribution buildDDist() {
        UniformDoubleDistribution eDist = new UniformDoubleDistribution();
        eDist.setStart(0.5);
        eDist.setEnd(1.0);
        return eDist;
    }

    public ParamSpace buildLParams() {
        ParamMap lParams = new ParamMap();
        lParams.add(LCSSDistance.EPSILON_FLAG, buildEDist());
        lParams.add(WINDOW_FLAG, buildDDist());
        return new ParamSpace(lParams);
    }

    public DiscreteParamDimension<DistanceMeasure> buildWDmParams() {
        DiscreteParamDimension<DistanceMeasure> wDmParams = new DiscreteParamDimension<>(
            Arrays.asList(new DTWDistance(), newDDTWDistance()));
        wDmParams.setSubSpace(buildWParams());
        return wDmParams;
    }

    public DiscreteParamDimension<DistanceMeasure> buildLDmParams() {
        DiscreteParamDimension<DistanceMeasure> lDmParams = new DiscreteParamDimension<>(
            Arrays.asList(new LCSSDistance()));
        lDmParams.setSubSpace(buildLParams());
        return lDmParams;
    }

    public ParamSpace buildParams() {
        ParamMap params = new ParamMap();
        params.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, lDmParams);
        params.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, wDmParams);
        return new ParamSpace(params);
    }

    @Test
    public void testAddAndGetForListOfValues() {
        List<ParamDimension<?>> valuesOut = wParams.getSingle().get(WINDOW_FLAG);
        List<?> values = ((DiscreteParamDimension<?>) valuesOut.get(0)).getValues();
        Assert.assertEquals(values, wParamValues);
    }

    @Test
    public void testAddAndGetForDistributionOfValues() {
        List<ParamDimension<?>> dimensions = lParams.getSingle().get(LCSSDistance.EPSILON_FLAG);
        for(ParamDimension<?> d : dimensions) {
            ContinuousParamDimension<?> dimension = (ContinuousParamDimension<?>) d;
            Object values = dimension.getDistribution();
            Assert.assertEquals(eDist, values);
        }
        dimensions = lParams.getSingle().get(WINDOW_FLAG);
        for(ParamDimension<?> d : dimensions) {
            ContinuousParamDimension<?> dimension = (ContinuousParamDimension<?>) d;
            Object values = dimension.getDistribution();
            Assert.assertEquals(values, dDist);
        }
    }

    @Test
    public void testParamsToString() {
//        System.out.println(params.toString());
        Assert.assertEquals("[{d=[{values=[LCSSDistance -w 1.0 -e 0.01], "
            + "subSpace=[{e=[dist=UniformDouble(0.0, 0.25)], "
            + "w=[dist=UniformDouble(0.5, 1.0)]}]}, {values=[DTWDistance -w 1.0, DDTWDistance -w 1.0], "
            + "subSpace=[{w=[{values=[1, 2, 3, 4, 5]}]}]}]}]", params.toString());
    }

    @Test
    public void testWParamsToString() {
//        System.out.println(wParams.toString());
        Assert.assertEquals("[{w=[{values=[1, 2, 3, 4, 5]}]}]", wParams.toString());
    }

    @Test
    public void testLParamsToString() {
//        System.out.println(lParams.toString());
        Assert.assertEquals("[{e=[dist=UniformDouble(0.0, 0.25)], "
            + "w=[dist=UniformDouble(0.5, 1.0)]}]", lParams.toString());
    }

    @Test
    public void testWDmParamsToString() {
//        System.out.println(wDmParams.toString());
        Assert.assertEquals("{values=[DTWDistance -w 1.0, DDTWDistance -w 1.0], subSpace=[{w=[{values=[1, "
            + "2, 3, 4, 5]}]}]}", wDmParams.toString());
    }

    @Test
    public void testLDmParamsToString() {
//        System.out.println(lDmParams.toString());
        Assert.assertEquals("{values=[LCSSDistance -w 1.0 -e 0.01], "
            + "subSpace=[{e=[dist=UniformDouble(0.0, 0.25)], "
            + "w=[dist=UniformDouble(0.5, 1.0)]}]}", lDmParams.toString());
    }

    @Test
    public void testEquals() {
        ParamSpace a = wParams;
        ParamSpace b = wParams;
        ParamSpace c = lParams;
        ParamSpace d = lParams;
        Assert.assertEquals(a, b);
        Assert.assertEquals(c, d);
        Assert.assertNotEquals(a, c);
        Assert.assertNotEquals(a.hashCode(), c.hashCode());
        Assert.assertNotEquals(b, c);
        Assert.assertNotEquals(b.hashCode(), c.hashCode());
        Assert.assertNotEquals(a, d);
        Assert.assertNotEquals(a.hashCode(), d.hashCode());
        Assert.assertNotEquals(b, d);
        Assert.assertNotEquals(b.hashCode(), d.hashCode());
    }
}
