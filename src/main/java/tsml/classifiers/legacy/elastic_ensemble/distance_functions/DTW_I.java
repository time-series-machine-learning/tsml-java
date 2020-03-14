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
package tsml.classifiers.legacy.elastic_ensemble.distance_functions;

import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author ABostrom
 */



public class DTW_I extends DTW_DistanceBasic{
   
    
    public DTW_I(){}
    
    public DTW_I(Instances train){
        super(train);
        
         m_Data = null;
         m_Validated = true;
    }
    
    //DIRTY HACK TO MAKE IT WORK WITH kNN. because of relational attribute stuff.
    @Override
    protected void validate() {}
    
    @Override
    public void update(Instance ins) {}
    
    @Override
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats){
        return distance(first,second,cutOffValue);
    }
    @Override    
    public double distance(Instance first, Instance second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }
    
    @Override
    public double distance(Instance multiSeries1, Instance multiseries2, double cutoff){
        
        //split the instance.
        Instance[] multi1 = splitMultivariateInstance(multiSeries1);
        Instance[] multi2 = splitMultivariateInstance(multiseries2);

        //TODO: might need to normalise here.
        
        //pairwise compare and sum dtw measures.
        double cumulative_distance = 0;
        for(int i=0; i< multi1.length; i++){
            cumulative_distance += Math.sqrt(super.distance(multi1[i], multi2[i], cutoff));
        }
        
        return cumulative_distance;
    }

}
