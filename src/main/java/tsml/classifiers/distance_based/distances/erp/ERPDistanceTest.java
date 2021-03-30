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
 
package tsml.classifiers.distance_based.distances.erp;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTWDistanceTest;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.core.Instances;

import static tsml.classifiers.distance_based.distances.dtw.spaces.DDTWDistanceSpace.newDDTWDistance;

public class ERPDistanceTest {
    private Instances instances;
    private ERPDistance df;

    @Before
    public void before() {
        instances = DTWDistanceTest.buildInstances();
        df = new ERPDistance();
        df.buildDistanceMeasure(instances);
    }

    @Test
    public void testFullWarpA() {
        df.setWindow(1);
        df.setG(1.5);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 182, 0);
    }

    @Test
    public void testFullWarpB() {
        df.setWindow(1);
        df.setG(2);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 175, 0);
    }

    @Test
    public void testConstrainedWarpA() {
        df.setWindow(0.2);
        df.setG(1.5);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 189.5, 0);
    }

    @Test
    public void testConstrainedWarpB() {
        df.setWindow(1);
        df.setG(2);
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 175, 0);
    }

}
