/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka.measures;

import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.elastic_distance_measures.DTW_DistanceBasic;
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
