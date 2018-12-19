/*
@inproceedings{rakthanmanon13fastshapelets,
author="T. Rakthanmanon and E. Keogh ",
title="Fast-Shapelets: A Fast Algorithm for Discovering Robust Time Series Shapelets",
booktitle    ="Proc. 13th {SDM}",
year="2013"
}

 */
package timeseriesweka.classifiers;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import static utilities.GenericTools.cloneArrayList;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/**
 *
 * @author raj09hxu
 */
public class FastShapelets extends AbstractClassifierWithTrainingData {

      
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.CONFERENCE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "T. Rakthanmanon and E. Keogh");
        result.setValue(TechnicalInformation.Field.TITLE, "Fast-Shapelets: A Fast Algorithm for Discovering Robust Time Series Shapelets");

        result.setValue(TechnicalInformation.Field.JOURNAL, "Proc. 13th SDM");
        result.setValue(TechnicalInformation.Field.YEAR, "2013");
        return result;
    }    
      
    

    
    static final int EXTRA_TREE_DEPTH = 2;
    static final float MIN_PERCENT_OBJ_SPLIT = 0.1f;
    static final float MAX_PURITY_SPLIT = 0.90f;
    static final int SH_MIN_LEN = 5;

    int MIN_OBJ_SPLIT;

    int numClass, numObj, subseqLength;

    int[] classFreq, orgClassFreq;
    ArrayList<ArrayList<Double>> orgData, data;
    ArrayList<Integer> Org_Label, Label;

    ArrayList<Integer> classifyList;

    ArrayList<Shapelet> finalSh;

    ArrayList<Pair<Integer, Double>> scoreList;

    //USAX_Map_Type is typedef unordered_map<SAX_word_type, USAX_elm_type> USAX_Map_type;
    //where a SAX_word_type is just an int.
    HashMap<Integer, USAX_elm_type> uSAXMap;

    private int seed;
    Random rand;

    //Obj_list_type  is a vector of ints. IE an ArrayList.
    // Node_Obj_set_type == vector<Obj_list_type> and Obj_list_type == vectorc<int>.. vector<vector<int>>
    ArrayList<ArrayList<Integer>> nodeObjList;

    double classEntropy;

    NN_ED nn;

    public FastShapelets() {
        nn = new NN_ED();
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainResults.buildTime=System.currentTimeMillis();
        train(data, 10, 10);
        trainResults.buildTime=System.currentTimeMillis()-trainResults.buildTime;
    }
    @Override
    public String getParameters() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getParameters());
//Add other fast shapelet parameters,         
        
        return sb.toString();
    }

    public void train(Instances data, int R, int top_k) {
        int sax_max_len, sax_len, w;
        int max_len = data.numAttributes() - 1, min_len = 10, step = 1; //consider whole search space.

        double percent_mask;
        Shapelet sh;

        rand = new Random(seed);

        numClass = data.numClasses();
        numObj = data.numInstances();

        sax_max_len = 15;
        percent_mask = 0.25;
        //R = 10;
        //top_k = 10;

        readTrainData(data);

        //initialise our data structures.
        nodeObjList = new ArrayList<>();
        finalSh = new ArrayList<>();
        uSAXMap = new HashMap<>();
        scoreList = new ArrayList<>();
        classifyList = new ArrayList<>();

        /// Find Shapelet
        for (int node_id = 1; (node_id == 1) || (node_id < nodeObjList.size()); node_id++) {
            Shapelet bsf_sh = new Shapelet();
            if (node_id <= 1) {
                setCurData(node_id);
            } else if (classifyList.get(node_id) == -1) { /// non-leaf node (-1:body node, -2:unused node)
                setCurData(node_id);
            } else {
                continue;
            }

            //3 to series length.
            for (subseqLength = min_len; subseqLength <= max_len; subseqLength += step) {
                /// Shapelet cannot be too short, e.g. len=1.
                if (subseqLength < SH_MIN_LEN) {
                    continue;
                }

                sax_len = sax_max_len;
                /// Make w and sax_len both integer
                w = (int) Math.ceil(1.0 * subseqLength / sax_len);
                sax_len = (int) Math.ceil(1.0 * subseqLength / w);

                createSAXList(subseqLength, sax_len, w);

                randomProjection(R, percent_mask, sax_len);
                scoreAllSAX(R);

                sh = findBestSAX(top_k);

                if (bsf_sh.lessThan(sh)) {
                    bsf_sh = sh;
                }

                uSAXMap.clear();
                scoreList.clear();
            }

            if (bsf_sh.len > 0) {
                double[] query = new double[bsf_sh.len];
                for (int i = 0; i < bsf_sh.len; i++) {
                    query[i] = this.data.get(bsf_sh.obj).get(bsf_sh.pos + i);
                }

                bsf_sh.setTS(query);
                finalSh.add(bsf_sh);
                /// post-processing: create tree
                setNextNodeObj(node_id, bsf_sh);
            }
        }
    }

    /// From top-k-score SAX
    /// Calculate Real Infomation Gain
    //
    Shapelet findBestSAX(int top_k) {
        //init the ArrayList with nulls.
        ArrayList<Pair<Integer, Double>> Dist = new ArrayList<>();
        for (int i = 0; i < numObj; i++) {
            Dist.add(null);
        }

        int word;
        double gain, dist_th, gap;
        int q_obj, q_pos;
        USAX_elm_type usax;
        int label, kk, total_c_in, num_diff;

        Shapelet sh = new Shapelet(), bsf_sh = new Shapelet();

        if (top_k > 0) {
            Collections.sort(scoreList, new ScoreComparator());
        }
        top_k = Math.abs(top_k);

        for (int k = 0; k < Math.min(top_k, scoreList.size()); k++) {
            word = scoreList.get(k).first;
            usax = uSAXMap.get(word);
            for (kk = 0; kk < Math.min(usax.sax_id.size(), 1); kk++) {
                int[] c_in = new int[numClass];
                int[] c_out = new int[numClass];
                //init the array list with 0s
                double[] query = new double[subseqLength];

                q_obj = usax.sax_id.get(kk).first;
                q_pos = usax.sax_id.get(kk).second;

                for (int i = 0; i < numClass; i++) {
                    c_in[i] = 0;
                    c_out[i] = classFreq[i];
                }
                for (int i = 0; i < subseqLength; i++) {
                    query[i] = data.get(q_obj).get(q_pos + i);
                }

                double dist;
                int m = query.length;
                double[] Q = new double[m];
                int[] order = new int[m];
                for (int obj = 0; obj < numObj; obj++) {
                    dist = nn.nearestNeighborSearch(query, data.get(obj), obj, Q, order);
                    Dist.set(obj, new Pair<>(obj, dist));
                }

                Collections.sort(Dist, new DistComparator());

                total_c_in = 0;
                for (int i = 0; i < Dist.size() - 1; i++) {
                    Pair<Integer, Double> pair_i = Dist.get(i);
                    Pair<Integer, Double> pair_ii = Dist.get(i + 1);

                    dist_th = (pair_i.second + pair_ii.second) / 2.0;
                    //gap = Dist[i+1].second - dist_th;
                    gap = ((double) (pair_ii.second - dist_th)) / Math.sqrt(subseqLength);
                    label = Label.get(pair_i.first);
                    c_in[label]++;
                    c_out[label]--;
                    total_c_in++;
                    num_diff = Math.abs(numObj - 2 * total_c_in);
                    //gain = CalInfoGain1(c_in, c_out);
                    gain = calcInfoGain2(c_in, c_out, total_c_in, numObj - total_c_in);

                    sh.setValueFew(gain, gap, dist_th);
                    if (bsf_sh.lessThan(sh)) {
                        bsf_sh.setValueAll(gain, gap, dist_th, q_obj, q_pos, subseqLength, num_diff, c_in, c_out);
                    }
                }
            }
        }
        return bsf_sh;
    }

    double calcInfoGain2(int[] c_in, int[] c_out, int total_c_in, int total_c_out) {
        return classEntropy - ((double) (total_c_in) / numObj * entropyArray(c_in, total_c_in) + (double) (total_c_out) / numObj * entropyArray(c_out, total_c_out));
    }

    /// Score each SAX
    void scoreAllSAX(int R) {
        int word;
        double score;
        USAX_elm_type usax;

        for (Map.Entry<Integer, USAX_elm_type> entry : uSAXMap.entrySet()) {
            word = entry.getKey();
            usax = entry.getValue();
            score = calcScore(usax, R);
            scoreList.add(new Pair<>(word, score));
        }
    }

    /// ***Calc***
    double calcScore(USAX_elm_type usax, int R) {
        double score = -1;
        int cid, count;
        double[] c_in = new double[numClass];       // Count object inside hash bucket
        double[] c_out = new double[numClass];      // Count object outside hash bucket

        /// Note that if no c_in, then no c_out of that object
        for (Map.Entry<Integer, Integer> entry : usax.obj_count.entrySet()) {
            cid = Label.get(entry.getKey());
            count = entry.getValue();
            c_in[cid] += (count);
            c_out[cid] += (R - count);
        }
        score = calcScoreFromObjCount(c_in, c_out);
        return score;
    }

    /// Score each sax in the matrix
    double calcScoreFromObjCount(double[] c_in, double[] c_out) {
        /// multi-class
        double diff, sum = 0, max_val = Double.NEGATIVE_INFINITY, min_val = Double.POSITIVE_INFINITY;
        for (int i = 0; i < numClass; i++) {
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

    /// Count the number of occurrences
    void randomProjection(int R, double percent_mask, int sax_len) {
        HashMap<Integer, HashSet<Integer>> Hash_Mark = new HashMap<>();
        int word, mask_word, new_word;
        HashSet<Integer> obj_set, ptr;

        int num_mask = (int) Math.ceil(percent_mask * sax_len);

        for (int r = 0; r < R; r++) {
            mask_word = createMaskWord(num_mask, sax_len);

            /// random projection and mark non-duplicate object
            for (Map.Entry<Integer, USAX_elm_type> entry : uSAXMap.entrySet()) {
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
            for (Map.Entry<Integer, USAX_elm_type> entry : uSAXMap.entrySet()) {
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

    /// create mask word (two random may give same position, we ignore it)
    int createMaskWord(int num_mask, int word_len) {
        int a, b;

        a = 0;
        for (int i = 0; i < num_mask; i++) {
            b = 1 << (rand.nextInt()%word_len); //generate a random number between 0 and the word_len
            a = a | b;
        }
        return a;
    }

    /// Set variables for next node. They are data, Label, classFreq, numObj
    void setCurData(int node_id) {
        if (node_id == 1) {
            //clone the arrayList
            data = new ArrayList<>();
            for (ArrayList<Double> a : orgData) {
                data.add(cloneArrayList(a));
            }
            Label = cloneArrayList(Org_Label);

            //clone the frequnecy array.
            classFreq = new int[orgClassFreq.length];
            System.arraycopy(orgClassFreq, 0, classFreq, 0, orgClassFreq.length);

        } else {
            ArrayList<Integer> it = nodeObjList.get(node_id);
            numObj = it.size();

            data.clear();
            Label.clear();

            for (int i = 0; i < numClass; i++) {
                classFreq[i] = 0;
            }

            int cur_class;

            //build our data structures based on the node and the labels and histogram.
            for (Integer in : it) {
                cur_class = Org_Label.get(in);
                data.add(orgData.get(in));
                Label.add(cur_class);
                classFreq[cur_class]++;
            }
        }
        classEntropy = entropyArray(classFreq, numObj);
    }

    /// new function still in doubt (as in Mueen's paper)
    double entropyArray(int[] A, int total) {
        double en = 0;
        double a;
        for (int i = 0; i < numClass; i++) {
            a = (double) A[i] / (double) total;
            if (a > 0) {
                en -= a * Math.log(a);
            }
        }
        return en;
    }

    void readTrainData(Instances data) {
        orgData = InstanceTools.fromWekaInstancesList(data);
        orgClassFreq = new int[numClass];
        Org_Label = new ArrayList<>();
        for (Instance i : data) {
            Org_Label.add((int) i.classValue());

            orgClassFreq[(int) i.classValue()]++;
        }
    }

    /// Fix card = 4 here !!!
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

    void createSAXList(int subseq_len, int sax_len, int w) {
        double ex, ex2, mean, std;
        double sum_segment[] = new double[sax_len];
        int elm_segment[] = new int[sax_len];
        int series, j, j_st, k, slot;
        double d;
        int word, prev_word;
        USAX_elm_type ptr;

        //init the element segments to the W value.
        for (k = 0; k < sax_len; k++) {
            elm_segment[k] = w;
        }

        elm_segment[sax_len - 1] = subseq_len - (sax_len - 1) * w;

        for (series = 0; series < data.size(); series++) {
            ex = ex2 = 0;
            prev_word = -1;

            for (k = 0; k < sax_len; k++) {
                sum_segment[k] = 0;
            }

            /// Case 1: Initial
            for (j = 0; (j < data.get(series).size()) && (j < subseq_len); j++) {
                d = data.get(series).get(j);
                ex += d;
                ex2 += d * d;
                slot = (int) Math.floor((j) / w);
                sum_segment[slot] += d;
            }

            /// Case 2: Slightly Update
            for (; (j <= (int) data.get(series).size()); j++) {

                j_st = j - subseq_len;
                mean = ex / subseq_len;
                std = Math.sqrt(ex2 / subseq_len - mean * mean);

                /// Create SAX from sum_segment
                word = createSAXWord(sum_segment, elm_segment, mean, std, sax_len);

                if (word != prev_word) {
                    prev_word = word;
                    //we're updating the reference so no need to re-add.
                    ptr = uSAXMap.get(word);
                    if (ptr == null) {
                        ptr = new USAX_elm_type();
                    }
                    ptr.obj_set.add(series);
                    ptr.sax_id.add(new Pair<>(series, j_st));
                    uSAXMap.put(word, ptr);
                }

                /// For next update
                if (j < data.get(series).size()) {
                    double temp = data.get(series).get(j_st);

                    ex -= temp;
                    ex2 -= temp * temp;

                    for (k = 0; k < sax_len - 1; k++) {
                        sum_segment[k] -= data.get(series).get(j_st + (k) * w);
                        sum_segment[k] += data.get(series).get(j_st + (k + 1) * w);
                    }
                    sum_segment[k] -= data.get(series).get(j_st + (k) * w);
                    sum_segment[k] += data.get(series).get(j_st + Math.min((k + 1) * w, subseq_len));

                    d = data.get(series).get(j);
                    ex += d;
                    ex2 += d * d;
                }
            }
        }
    }

    void setNextNodeObj(int node_id, Shapelet sh) {
        int q_obj = sh.obj;
        int q_pos = sh.pos;
        int q_len = sh.len;
        double dist_th = sh.dist_th;
        double[] query = new double[q_len];

        int left_node_id = node_id * 2;
        int right_node_id = node_id * 2 + 1;
        int real_obj;

        /// Memory Allocation
        while (nodeObjList.size() <= right_node_id) {
            nodeObjList.add(new ArrayList<Integer>());
            classifyList.add(-2);
            finalSh.add(new Shapelet());

            if (nodeObjList.size() == 2) {   /// Note that nodeObjList[0] is not used
                for (int i = 0; i < numObj; i++) {
                    nodeObjList.get(1).add(i);
                }
            }
        }

        finalSh.set(node_id, sh);

        /// Use the shapelet on previous data
        for (int i = 0; i < q_len; i++) {
            query[i] = data.get(q_obj).get(q_pos + i);
        }

        double dist;
        int m = query.length;
        double[] Q = new double[m];
        int[] order = new int[m];

        for (int obj = 0; obj < numObj; obj++) {
            dist = nn.nearestNeighborSearch(query, data.get(obj), obj, Q, order);
            real_obj = nodeObjList.get(node_id).get(obj);
            int node = dist <= dist_th ? left_node_id : right_node_id; //left or right node?
            nodeObjList.get(node).add(real_obj);
        }
        /// If left/right is pure, or so small, stop spliting
        int max_c_in = -1, sum_c_in = 0;
        int max_c_out = -1, sum_c_out = 0;
        int max_ind_c_in = -1, max_ind_c_out = -1;
        for (int i = 0; i < sh.c_in.length; i++) {
            int c_in_i = sh.c_in[i];
            int c_out_i = sh.c_out[i];

            sum_c_in += c_in_i;
            if (max_c_in < c_in_i) {
                max_c_in = c_in_i;
                max_ind_c_in = i;
            }

            sum_c_out += c_out_i;
            if (max_c_out < c_out_i) {
                max_c_out = c_out_i;
                max_ind_c_out = i;
            }
        }

        boolean left_is_leaf = false;
        boolean right_is_leaf = false;

        MIN_OBJ_SPLIT = (int) Math.ceil((double) (MIN_PERCENT_OBJ_SPLIT * numObj) / (double) numClass);
        if ((sum_c_in <= MIN_OBJ_SPLIT) || ((double) max_c_in / (double) sum_c_in >= MAX_PURITY_SPLIT)) {
            left_is_leaf = true;
        }
        if ((sum_c_out <= MIN_OBJ_SPLIT) || ((double) max_c_out / (double) sum_c_out >= MAX_PURITY_SPLIT)) {
            right_is_leaf = true;
        }

        int max_tree_dept = (int) (EXTRA_TREE_DEPTH + Math.ceil(Math.log(numClass) / Math.log(2)));
        if (node_id >= Math.pow(2, max_tree_dept)) {
            left_is_leaf = true;
            right_is_leaf = true;
        }

        //set node.
        classifyList.set(node_id, -1);

        //set left child.
        int val = left_is_leaf ? max_ind_c_in : -1;
        classifyList.set(left_node_id, val);

        //set right child.
        val = right_is_leaf ? max_ind_c_out : -1;
        classifyList.set(right_node_id, val);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        int node_id, m;
        double d, dist_th;

        double[] dArray = instance.toDoubleArray();
        ArrayList<Double> data = new ArrayList<>();
        //-1 off length so we don't add the classValue.
        for (int i = 0; i < dArray.length - 1; i++) {
            data.add(dArray[i]);
        }

        int tree_size = nodeObjList.size();

        /// start at the top node
        node_id = 1;
        while ((classifyList.get(node_id) < 0) || (node_id > tree_size)) {
            Shapelet node = finalSh.get(node_id);

            m = node.len;
            double[] Q = new double[m];
            int[] order = new int[m];

            d = nn.nearestNeighborSearch(node.ts, data, 0, Q, order);
            dist_th = node.dist_th;

            if (d <= dist_th) {
                node_id = 2 * node_id;
            } else {
                node_id = 2 * node_id + 1;
            }

        }

        return (double) classifyList.get(node_id);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double c=classifyInstance(instance);
        double[] r=new double[instance.numClasses()];
        r[(int)c]=1;
        return r;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void main(String[] args) throws Exception {
        final String dotdotSlash = ".." + File.separator;
        String datasetName = "ItalyPowerDemand";
        String datasetLocation = dotdotSlash + dotdotSlash + "resampled data sets" + File.separator + datasetName + File.separator + datasetName;

        for (int i = 0; i < 100; i++) {
            Instances train = utilities.ClassifierTools.loadData(datasetLocation + i + "_TRAIN");
            Instances test = utilities.ClassifierTools.loadData(datasetLocation + i + "_TEST");

            FastShapelets fs = new FastShapelets();

            try {
                fs.buildClassifier(train);

                double accuracy = utilities.ClassifierTools.accuracy(test, fs);
                System.out.println("fold " + i + " acc: " + accuracy);

            } catch (Exception ex) {
                System.out.println("Exception " + ex);
            }
        }
    }

    /**
     * @param seed the seed to set
     */
    public void setSeed(int seed) {
        this.seed = seed;
    }

    private class ScoreComparator implements Comparator<Pair<Integer, Double>> {

        @Override
        //if the left one is bigger put it closer to the top.
        public int compare(Pair<Integer, Double> t, Pair<Integer, Double> t1) {
            return Double.compare(t1.second, t.second);
        }

    }

    private class DistComparator implements Comparator<Pair<Integer, Double>> {

        @Override
        public int compare(Pair<Integer, Double> t, Pair<Integer, Double> t1) {
            return Double.compare(t.second, t1.second);
        }

    }

    private class Shapelet {

        public double gain;
        public double gap;
        public double dist_th;
        public int obj;
        public int pos;
        public int len;
        public int num_diff;
        int[] c_in;
        int[] c_out;
        double[] ts;

        public Shapelet() {
            gain = Double.NEGATIVE_INFINITY;
            gap = Double.NEGATIVE_INFINITY;
            dist_th = Double.POSITIVE_INFINITY;
            obj = -1;
            pos = -1;
            len = -1;
            num_diff = -1;
        }

        void setValueFew(double gain, double gap, double dist_th) {
            this.gain = gain;
            this.gap = gap;
            this.dist_th = dist_th;
        }

        void setValueAll(double gain, double gap, double dist_th, int obj, int pos, int len, int num_diff, int[] in, int[] out) {
            this.gain = gain;
            this.gap = gap;
            this.dist_th = dist_th;
            this.obj = obj;
            this.pos = pos;
            this.len = len;
            this.num_diff = num_diff;

            c_in = new int[in.length];
            c_out = new int[out.length];
            System.arraycopy(in, 0, c_in, 0, in.length);
            System.arraycopy(out, 0, c_out, 0, out.length);
        }

        void setTS(double[] ts) {
            this.ts = ts;
        }

        private boolean lessThan(Shapelet other) {
            if (gain > other.gain) {
                return false;
            }
            return ((gain < other.gain)
                    || ((gain == other.gain) && (num_diff > other.num_diff))
                    || ((gain == other.gain) && (num_diff == other.num_diff) && (gap < other.gap)));
        }
    }

    private class USAX_elm_type {

        HashSet<Integer> obj_set;
        ArrayList<Pair<Integer, Integer>> sax_id;
        HashMap<Integer, Integer> obj_count;

        public USAX_elm_type() {
            obj_set = new HashSet<>();
            sax_id = new ArrayList<>();
            obj_count = new HashMap<>();
        }

    }

    private class Pair<A, B> {

        public A first;
        public B second;

        Pair() {
        }

        Pair(A l, B r) {
            first = l;
            second = r;
        }
    }

    private class NN_ED {

        private class Index implements Comparable<Index> {

            double value;
            int index;

            public Index() {
            }

            @Override
            public int compareTo(Index t) {
                return Math.abs((int) this.value) - Math.abs((int) t.value);
            }
        }

        public NN_ED() {
        }

        double nearestNeighborSearch(double[] query, ArrayList<Double> data, int obj_id, double[] Q, int[] order) {
            double bsf;
            int m, M;
            double d;
            int i;
            int j;
            double ex, ex2, mean, std;
            int loc = 0;

            m = query.length;
            M = data.size();

            bsf = Double.MAX_VALUE;
            i = 0;
            j = 0;
            ex = ex2 = 0;

            if (obj_id == 0) {
                for (i = 0; i < m; i++) {
                    d = query[i];
                    ex += d;
                    ex2 += d * d;
                    Q[i] = d;
                }

                mean = ex / m;
                std = ex2 / m;
                std = Math.sqrt(std - mean * mean);

                for (i = 0; i < m; i++) {
                    Q[i] = (Q[i] - mean) / std;
                }

                Index[] Q_tmp = new Index[m];
                for (i = 0; i < m; i++) {
                    Q_tmp[i] = new Index();
                    Q_tmp[i].value = Q[i];
                    Q_tmp[i].index = i;
                }

                Arrays.sort(Q_tmp);
                for (i = 0; i < m; i++) {
                    Q[i] = Q_tmp[i].value;
                    order[i] = Q_tmp[i].index;
                }
            }

            i = 0;
            j = 0;
            ex = ex2 = 0;

            double[] T = new double[2 * m];

            double dist = 0;
            while (i < M) {
                d = data.get(i);
                ex += d;
                ex2 += d * d;
                T[i % m] = d;
                T[(i % m) + m] = d;

                if (i >= m - 1) {
                    mean = ex / m;
                    std = ex2 / m;
                    std = Math.sqrt(std - mean * mean);

                    j = (i + 1) % m;
                    dist = distance(Q, order, T, j, m, mean, std, bsf);
                    if (dist < bsf) {
                        bsf = dist;
                        loc = i - m + 1;
                    }
                    ex -= T[j];
                    ex2 -= T[j] * T[j];
                }
                i++;
            }
            return bsf;
        }

        double distance(double[] Q, int[] order, double[] T, int j, int m, double mean, double std, double best_so_far) {
            int i;
            double sum = 0;
            double bsf2 = best_so_far * best_so_far;
            for (i = 0; i < m && sum < bsf2; i++) {
                double x = (T[(order[i] + j)] - mean) / std;
                sum += (x - Q[i]) * (x - Q[i]);
            }
            return Math.sqrt(sum);
        }

        double distance(double[] Q, int[] order, double[] T, int j, int m, double mean, double std) {
            return distance(Q, order, T, j, m, mean, std, Double.MAX_VALUE);
        }
    }

}
