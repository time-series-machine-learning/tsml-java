/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.classifiers.multivariate;

import tsml.classifiers.legacy.elastic_ensemble.distance_functions.EuclideanDistance_D;
import static utilities.InstanceTools.findMinDistance;
import utilities.generic_storage.Pair;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Aaron
 */
public class NN_ED_D extends MultivariateAbstractClassifier{
    
    Instances train;
    EuclideanDistance_D D;
    
    public NN_ED_D(){
        D = new EuclideanDistance_D();
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        testWithFailRelationalInstances(data);
        train = data;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception{
        testWithFailRelationalInstance(instance);
        Pair<Instance, Double> minD = findMinDistance(train, instance, D);
        return minD.var1.classValue();
    }
}
