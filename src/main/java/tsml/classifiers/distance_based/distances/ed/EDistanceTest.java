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
 
package tsml.classifiers.distance_based.distances.ed;

import static tsml.classifiers.distance_based.distances.dtw.DTWDistanceTest.*;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import weka.core.Instances;

public class EDistanceTest {
    @Test
    public void matchesDtwZeroWindow() {
        DTWDistance dtw = new DTWDistance();
        dtw.setWindow(0);
        final Instances instances = buildInstances();
        dtw.buildDistanceMeasure(instances);
        final double d1 = df.distance(instances.get(0), instances.get(1));
        final double d2 = dtw.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(d1, d2, 0d);
    }

    private Instances instances;
    private EDistance df;

    @Before
    public void before() {
        instances = buildInstances();
        df = new EDistance();
        df.buildDistanceMeasure(instances);
    }


}
