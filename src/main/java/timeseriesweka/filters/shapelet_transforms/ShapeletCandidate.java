/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms;

import java.io.Serializable;

/**
 *
 * @author raj09hxu
 */
public class ShapeletCandidate implements Serializable{
    double[][] content;
    int numChannels;
    
    //if no dimension, assume univariate
    public ShapeletCandidate(){
        numChannels = 1;
        content = new double[numChannels][];
    }
    
    public ShapeletCandidate(int numChans){
        numChannels = numChans;
        content = new double[numChannels][];
    }
    
    public ShapeletCandidate(double[] cont){
        numChannels = 1;
        content = new double[numChannels][];
        content[0] = cont;
    }
    
    //if no dimension, assume univariate
    public void setShapeletContent(double[] cont){
        content[0] = cont;
    }
    
    public void setShapeletContent(int channel, double[] cont){
        content[channel] = cont;
    }

    public double[] getShapeletContent(int channel){
        return content[channel]; 
    }
    
    public double[] getShapeletContent(){
        return content[0];
    }
    
    public int getLength(){
        return content[0].length;
    }
    
    public int getNumChannels(){
        return numChannels;
    }
}
