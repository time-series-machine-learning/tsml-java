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
package timeseriesweka.filters.shapelet_transforms;

import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.TreeMap;
import weka.core.Instances;

/**
 * 
 * Tony's stripped down shapelet finder.
 */
public class ShapeletTransformLight implements Serializable
{
    private int numShapelets=1000;  //Max number of shapelets 
    private Instances trainData;       
    protected ArrayList<Shapelet> shapelets;//Final list of shapelets
    double allowedTime=1000*60*60;  //Default to one hour

    public void findBestShapelets(Instances data){
        double startTime=System.currentTimeMillis();
        double currentTime=System.currentTimeMillis();

        int c=data.numClasses();
        int m=data.numAttributes()-1;
        int n=data.numInstances();
        trainData=data;
        
        PriorityQueue<Shapelet>[] shapeletsByClass= new PriorityQueue[c];
        for (int i=0; i <c ; i++)
            shapeletsByClass[i] = new PriorityQueue<>();
//Store in a priority MAX queue with the worst at the top. 
//Once we have the minimum of each class, remove the worst so far if the new one is better
   //Number shapelets per class
        int shapeletsPerClass = numShapelets/c;
//Start the timer
        boolean timesUp=false;
        while(currentTime-startTime<allowedTime){
//sample shapelets for each class for time allocated to each class based 
            int caseIndex=sampleCase(data);
            Shapelet s= sampleShapelet(caseIndex);
            
//Assess shapelet quality
            
//Keep the best ones
//            if(shapeletsByClass[i].size()<shapeletsPerClass)
//                shapeletsByClass[i].add(s);
 //           else{
                //REmove worst then add
 //           }
        }
//when finished, merge into a single set of shapelets.         
 
    //    shapelets = mergeBestKShapelets(shapeletsByClass);
        
        this.numShapelets = shapelets.size();
    }
    private int sampleCase(Instances d){
        return 1;
    }
    private Shapelet sampleShapelet(int index){
        return null;
    }


    private ArrayList<Shapelet> mergeBestKShapelets(ArrayList<Shapelet>[] shapeletsByClass)
    {
       ArrayList<Shapelet> kShapelets = new ArrayList<>();
       
       int numberOfClassVals = shapeletsByClass.length;
       int proportion = numShapelets/numberOfClassVals;
       
       
       Iterator<Shapelet> it;
       
       //all lists should be sorted.
       //go through the map and get the sub portion of best shapelets for the final list.
       for(ArrayList<Shapelet> list : shapeletsByClass){
           int i=0;
           it = list.iterator();
           while(it.hasNext() && i++ <= proportion)
               kShapelets.add(it.next());
       }
       return kShapelets;
    }
    
    private class Shapelet implements Comparable<Shapelet>{
        int caseIndex;
        int startPosition;
        int length;
        double quality;

        @Override
        public int compareTo(Shapelet t) { //For min Heap, i.e. worst is next to process and has highest priority
            if(quality>t.quality) return 1;
            if(quality<t.quality) return -1;
            return 0;
        }
    }
}
