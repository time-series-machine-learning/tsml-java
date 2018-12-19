/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.search_functions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.filters.shapelet_transforms.Shapelet;

/**
 *
 * @author raj09hxu
 */

//only for Univariate
public class FastShapeletSearch extends ShapeletSearch implements Serializable{
    
    int R = 10;
    int sax_max_len = 15;
    double percent_mask = 0.25;
    long seed;
    Random rand;
    boolean searched = false;
    
    ArrayList<Pair<Integer, Double>> Score_List;
    HashMap<Integer, USAX_elm_type> USAX_Map;

    protected FastShapeletSearch(ShapeletSearchOptions ops) {
        super(ops);
    }
    
    @Override
    public void init(Instances input){
        super.init(input);
        searched = false;
    }
    
    @Override
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){

        int index = utilities.InstanceTools.indexOf(inputData, timeSeries);
        
        //becase the fast shapelets does all series rather than incrementally doing each series, we want to abandon further calls.
        //we index the shapelets in the series they come from, because then we can just grab them as a series is asked for.
        if(!searched){
            calculateShapelets();
            searched = true;
        }
        
        
        int word;
        int id, pos, len;
        USAX_elm_type usax;  

        Collections.sort(Score_List, new ScoreComparator());

        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        //for the top K SAX words.
        for (Pair<Integer, Double> Score_List1 : Score_List) {
            word = Score_List1.first;
            usax = USAX_Map.get(word);
            int kk;

            //get the first one out.
            //this is because some sax words represent multiple start positions.
            for (kk = 0; kk < Math.min(usax.sax_id.size(), 1); kk++) {
                id = usax.sax_id.get(kk).first;
                
                if(id != index) continue; //if the id doesn't match our current asked for one. ignore.
                
                pos = usax.sax_id.get(kk).second;
                len = usax.sax_id.get(kk).third;
                //init the array list with 0s
                Shapelet s =  checkCandidate.process(inputData.get(id), pos, len);
                if(s != null){
                    //put the shapelet in the list from it's series.
                    seriesShapelets.add(s);
                }
            }
        }
        
        //definitely think we can reduce the amount of work even more. 
        //by reducing seriesShapelets even more. Not letting it be more than K. etc.
          
        return seriesShapelets;
    }
        
    private void calculateShapelets(){
        for (int length = minShapeletLength; length <= maxShapeletLength; length+=lengthIncrement) {
            USAX_Map = new HashMap<>();
            Score_List = new ArrayList<>();

            int sax_len = sax_max_len;
            /// Make w and sax_len both integer
            int w = (int) Math.ceil(1.0 * length / sax_len);
            sax_len = (int) Math.ceil(1.0 * length / w);

            createSAXList(length, sax_len, w);

            randomProjection(R, percent_mask, sax_len);
            scoreAllSAX(R);
        }
    }

    void createSAXList(int subseq_len, int sax_len, int w) {
        double ex, ex2, mean, std;
        double sum_segment[] = new double[sax_len];
        int elm_segment[] = new int[sax_len];
        int series, j, j_st, k, slot;
        double d;
        int word, prev_word;
        int numAttributes = seriesLength-1;
        
        USAX_elm_type ptr;

        //init the element segments to the W value.
        for (k = 0; k < sax_len; k++) {
            elm_segment[k] = w;
        }

        elm_segment[sax_len - 1] = subseq_len - (sax_len - 1) * w;

        double[] timeSeries;
        
        for (series = 0; series < inputData.size(); series++) {
            timeSeries = inputData.get(series).toDoubleArray();
            
            
            ex = ex2 = 0;
            prev_word = -1;

            for (k = 0; k < sax_len; k++) {
                sum_segment[k] = 0;
            }

            // create first subsequence. PAA'ing as we go.
            for (j = 0; (j < numAttributes) && (j < subseq_len); j++) {
                d = timeSeries[j];
                ex += d;
                ex2 += d * d;
                slot = (int) Math.floor((j) / w);
                sum_segment[slot] += d;
            }

            /// Case 2: Slightly Update
            for (; j <= numAttributes; j++) {
                j_st = j - subseq_len;
                mean = ex / subseq_len;
                std = Math.sqrt(ex2 / subseq_len - mean * mean);

                /// Create SAX from sum_segment
                word = createSAXWord(sum_segment, elm_segment, mean, std, sax_len);

                if (word != prev_word) {
                    prev_word = word;
                    //we're updating the reference so no need to re-add.
                    ptr = USAX_Map.get(word);
                    if (ptr == null) {
                        ptr = new USAX_elm_type();
                    }
                    ptr.obj_set.add(series);
                    ptr.sax_id.add(new Triplet<>(series, j_st, subseq_len));
                    USAX_Map.put(word, ptr);
                }

                /// For next update
                if (j < numAttributes) {
                    double temp = timeSeries[j_st];

                    ex -= temp;
                    ex2 -= temp * temp;

                    for (k = 0; k < sax_len - 1; k++) {
                        sum_segment[k] -= timeSeries[j_st + (k) * w];
                        sum_segment[k] += timeSeries[j_st + (k + 1) * w];
                    }
                    sum_segment[k] -= timeSeries[j_st + (k) * w];
                    sum_segment[k] += timeSeries[j_st + Math.min((k + 1) * w, subseq_len)];

                    d = timeSeries[j];
                    ex += d;
                    ex2 += d * d;
                }
            }
        }
    }
    // Fix card = 4 here !!!
    //create a sax word of size 4 here as an int.
    int createSAXWord(double[] sum_segment, int[] elm_segment, double mean, double std, int sax_len) {
        int word = 0, val = 0;
        double d = 0;

        for (int i = 0; i < sax_len; i++) {
            d = (sum_segment[i] / elm_segment[i] - mean) / std;
            if (d < 0) {
                if (d < -0.67) {
                    val = 0;
                } else {
                    val = 1;
                }
            } else if (d < 0.67) {
                val = 2;
            } else {
                val = 3;
            }

            word = (word << 2) | (val);
        }
        return word;
    }
    
    // Count the number of occurrences
    void randomProjection(int R, double percent_mask, int sax_len) {
        HashMap<Integer, HashSet<Integer>> Hash_Mark = new HashMap<>();
        int word, mask_word, new_word;
        HashSet<Integer> obj_set, ptr;

        int num_mask = (int) Math.ceil(percent_mask * sax_len);

        for (int r = 0; r < R; r++) {
            mask_word = createMaskWord(num_mask, sax_len);

            /// random projection and mark non-duplicate object
            for (Map.Entry<Integer, USAX_elm_type> entry : USAX_Map.entrySet()) {
                word = entry.getKey();
                obj_set = entry.getValue().obj_set;

                //put the new word and set combo in the hash_mark
                new_word = word | mask_word;

                ptr = Hash_Mark.get(new_word);

                if (ptr == null) {
                    Hash_Mark.put(new_word, new HashSet<>(obj_set));
                } else {
                    //add onto our ptr, rather than overwrite.
                    ptr.addAll(obj_set);
                }
            }

            /// hash again for keep the count
            for (Map.Entry<Integer, USAX_elm_type> entry : USAX_Map.entrySet()) {
                word = entry.getKey();
                new_word = word | mask_word;
                obj_set = Hash_Mark.get(new_word);
                //increase the histogram
                for (Integer o_it : obj_set) {
                    Integer count = entry.getValue().obj_count.get(o_it);
                    count = count == null ? 1 : count + 1;
                    entry.getValue().obj_count.put(o_it, count);
                }
            }

            Hash_Mark.clear();
        }
    }
    
    // create mask word (two random may give same position, we ignore it)
    int createMaskWord(int num_mask, int word_len) {
        int a, b;

        a = 0;
        for (int i = 0; i < num_mask; i++) {
            b = 1 << (word_len / 2);
            //b = 1 << (rand.nextInt()%word_len); //generate a random number between 0 and the word_len
            a = a | b;
        }
        return a;
    }
    
    // Score each SAX
    void scoreAllSAX(int R) {
        int word;
        double score;
        USAX_elm_type usax;

        for (Map.Entry<Integer, USAX_elm_type> entry : USAX_Map.entrySet()) {
            word = entry.getKey();
            usax = entry.getValue();
            score = calcScore(usax, R);
            Score_List.add(new Pair<>(word, score));
        }
    }
    
    double calcScore(USAX_elm_type usax, int R) {
        double score = -1;
        int cid, count;
        double[] c_in = new double[inputData.numClasses()];       // Count object inside hash bucket
        double[] c_out = new double[inputData.numClasses()];      // Count object outside hash bucket

        /// Note that if no c_in, then no c_out of that object
        for (Map.Entry<Integer, Integer> entry : usax.obj_count.entrySet()) {
            cid = (int) inputData.get(entry.getKey()).classValue();
            count = entry.getValue();
            c_in[cid] += (count);
            c_out[cid] += (R - count);
        }
        score = calcScoreFromObjCount(c_in, c_out);
        return score;
    }
    
    // Score each sax in the matrix
    double calcScoreFromObjCount(double[] c_in, double[] c_out) {
        /// multi-class
        double diff, sum = 0, max_val = Double.NEGATIVE_INFINITY, min_val = Double.POSITIVE_INFINITY;
        for (int i = 0; i < inputData.numClasses(); i++) {
            diff = (c_in[i] - c_out[i]);
            if (diff > max_val) {
                max_val = diff;
            }
            if (diff < min_val) {
                min_val = diff;
            }
            sum += Math.abs(diff);
        }
        return (sum - Math.abs(max_val) - Math.abs(min_val)) + Math.abs(max_val - min_val);
    }

    private class ScoreComparator implements Comparator<Pair<Integer, Double>>, Serializable{

        @Override
        //if the left one is bigger put it closer to the top.
        public int compare(Pair<Integer, Double> t, Pair<Integer, Double> t1) {
            return Double.compare(t1.second, t.second);
        }

    }

    private class USAX_elm_type implements Serializable{

        HashSet<Integer> obj_set;
        ArrayList<Triplet<Integer, Integer, Integer>> sax_id;
        HashMap<Integer, Integer> obj_count;

        public USAX_elm_type() {
            obj_set = new HashSet<>();
            sax_id = new ArrayList<>();
            obj_count = new HashMap<>();
        }

    }

    private class Pair<A, B> implements Serializable{

        public A first;
        public B second;

        Pair() {
        }

        Pair(A l, B r) {
            first = l;
            second = r;
        }
    }
    
       private class Triplet<A, B, C> implements Serializable{

        public A first;
        public B second;
        public C third;

        Triplet() {
        }

        Triplet(A l, B r, C g) {
            first = l;
            second = r;
            third = g;
        }
    }
    
}
