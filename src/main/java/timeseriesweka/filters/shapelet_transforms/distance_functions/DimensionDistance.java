/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.distance_functions;

import java.io.Serializable;
import java.util.Arrays;
import weka.core.Instance;

/**
 *
 * @author raj09hxu
 */
public class DimensionDistance extends ImprovedOnlineSubSeqDistance implements Serializable{
    
    @Override
    public double calculate(Instance timeSeries, int timeSeriesId){
        //split the timeSeries up and pass in the specific shapelet dim.
        Instance[] dimensions = utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance(timeSeries);
        return calculate(dimensions[dimension].toDoubleArray(), timeSeriesId);
    }
    
}
