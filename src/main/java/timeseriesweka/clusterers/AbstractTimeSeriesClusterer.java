/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.clusterers;

import weka.clusterers.AbstractClusterer;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Abstract class for time series based clusterers.
 *
 * @author pfm15hbu
 */
public abstract class AbstractTimeSeriesClusterer extends AbstractClusterer{
    
    protected boolean copyInstances = true;

    protected int[] assignments;
    protected ArrayList<Integer>[] clusters;

    public int[] getAssignments(){
        return assignments;
    }

    public ArrayList<Integer>[] getClusters(){
        return clusters;
    }
    
    public void setCopyInstances(boolean b){
        copyInstances = b;
    }
}
