/*

Wrapper for a DTW implementation that speeds up the window size search
through caching

 */
package timeseriesweka.classifiers.distance_based;


import java.util.ArrayList;
import evaluation.storage.ClassifierResults;
import java.util.concurrent.TimeUnit;
import timeseriesweka.classifiers.SaveParameterInfo;
import timeseriesweka.classifiers.distance_based.fast_dtw.windowSearcher.FastWWSByPercent;
import timeseriesweka.classifiers.distance_based.fast_dtw.windowSearcher.WindowSearcher;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

/**
 *
 * @author ajb
 * 
 * Wrapper for Fast DTW Window Search 
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" 
 * published in SDM18
 * 
 * Search for the best warping window using Fast Warping Window Search (FastWWS)
 * 
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb


*/
public class FastDTW extends AbstractClassifier  implements SaveParameterInfo, TechnicalInformationHandler{

    WindowSearcher ws;
    protected ArrayList<Double> buildTimes;
    protected ClassifierResults res =new ClassifierResults();
    
    public FastDTW(){
        ws=new FastWWSByPercent();
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        res.setTimeUnit(TimeUnit.NANOSECONDS);
        long startTime=System.nanoTime(); 
        ws.buildClassifier(data);
        res.setBuildTime(System.nanoTime()-startTime);
        Runtime rt = Runtime.getRuntime();
        long usedBytes = (rt.totalMemory() - rt.freeMemory());
        res.setMemory(usedBytes);
    }
    public double classifyInstance(Instance data) throws Exception {
       return ws.classifyInstance(data);
    }

    @Override
    public String getParameters() {
        String result="CVAcc,"+res.getAcc()+",Memory,"+res.getMemory();
        result+=",WindowSize,"+ws.getBestWin()+",Score,"+ws.getBestScore();
        return result;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.CONFERENCE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier and Geoff Webb");
        result.setValue(TechnicalInformation.Field.YEAR, "2018");
        result.setValue(TechnicalInformation.Field.TITLE, "Efficient search of the best warping window for Dynamic Time Warping");
        result.setValue(TechnicalInformation.Field.HOWPUBLISHED," Proceeding of Siam Data Mining Conference");
        return result;
    }

    
}
