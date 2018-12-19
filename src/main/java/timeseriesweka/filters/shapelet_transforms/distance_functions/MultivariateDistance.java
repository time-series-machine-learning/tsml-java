/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.distance_functions;

import java.io.Serializable;
import timeseriesweka.filters.shapelet_transforms.ShapeletCandidate;
import static utilities.multivariate_tools.MultivariateInstanceTools.convertMultiInstanceToArrays;
import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class MultivariateDistance extends SubSeqDistance implements Serializable{
    protected int numChannels;
    protected int seriesLength;
    
    @Override
    public void init(Instances data)
    {
        count =0;
        numChannels = utilities.multivariate_tools.MultivariateInstanceTools.numChannels(data);
        seriesLength = utilities.multivariate_tools.MultivariateInstanceTools.channelLength(data);
        
    }
    
    protected double[][] candidateArray2;
    
    @Override
    public void setCandidate(Instance inst, int start, int len, int dim) {
        //extract shapelet and nomrliase.
        cand = new ShapeletCandidate(numChannels);
        startPos = start;
        length = len;
        
        //only call to double array when we've changed series.
        if(candidateInst==null || candidateInst != inst){
            candidateArray2 = convertMultiInstanceToArrays(splitMultivariateInstance(inst));
            candidateInst = inst;
        }
        
        for(int i=0; i< numChannels; i++){
            double[] temp = new double[length];
            //copy the data from the whole series into a candidate.
            System.arraycopy(candidateArray2[i], start, temp, 0, length);
            temp = zNormalise(temp, false); //normalise each series.
            cand.setShapeletContent(i, temp);
        } 
    }    
}
