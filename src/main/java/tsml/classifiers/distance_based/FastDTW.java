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
 

/*

Wrapper for a DTW implementation that speeds up the window size search
through caching

 */
package tsml.classifiers.distance_based;


import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.legacy.elastic_ensemble.fast_window_search.windowSearcher.FastWWSByPercent;
import tsml.classifiers.legacy.elastic_ensemble.fast_window_search.windowSearcher.WindowSearcher;
import weka.core.*;

/**
 * Wrapper for Chan Wei's implementation of DTW
 * @author ajb
 */
public class FastDTW extends EnhancedAbstractClassifier{

    WindowSearcher ws;
    protected ArrayList<Double> buildTimes;
    
    public FastDTW(){
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        ws=new FastWWSByPercent();
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        long startTime=System.nanoTime(); 
        ws.buildClassifier(data);
        trainResults.setBuildTime(System.nanoTime()-startTime);
        Runtime rt = Runtime.getRuntime();
        long usedBytes = (rt.totalMemory() - rt.freeMemory());
        trainResults.setMemory(usedBytes);
    }
    public double classifyInstance(Instance data) throws Exception {
       return ws.classifyInstance(data);
    }

    @Override
    public String getParameters() {
        String result="CVAcc,"+trainResults.getAcc()+",Memory,"+trainResults.getMemory();
        result+=",WindowSize,"+ws.getBestWin()+",Score,"+ws.getBestScore();
        return result;
    }

    
}
