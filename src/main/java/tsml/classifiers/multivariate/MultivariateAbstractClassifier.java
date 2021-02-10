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
 
package tsml.classifiers.multivariate;

//import tsml.classifiers.distance_based.distances.old.DTW_D;
import utilities.generic_storage.Pair;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import static utilities.InstanceTools.findMinDistance;


/**
 *
 * @author Alejandro Pasos Ruiz
 */
public abstract class MultivariateAbstractClassifier extends AbstractClassifier {

    public MultivariateAbstractClassifier(){
        super();
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.disable(Capabilities.Capability.MISSING_VALUES);
        return result;
    }

    protected void testWithFailRelationalInstances(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        for (Instance instance: data){
            testWithFailRelationalInstance(instance);
        }

    }

    protected void testWithFailRelationalInstance(Instance data) throws Exception {
            Instances group = MultivariateInstanceTools.splitMultivariateInstanceOnInstances(data);
            getCapabilities().testWithFail(group);
    }




}
