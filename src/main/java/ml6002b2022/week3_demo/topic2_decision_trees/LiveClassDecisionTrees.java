package ml6002b2022.week3_demo.topic2_decision_trees;

import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.InfoGainSplitCrit;
import weka.core.Instances;

import java.text.DecimalFormat;

public class LiveClassDecisionTrees {
    static String dataPath="C:\\Users\\Tony\\OneDrive - University of East Anglia\\Teaching\\2020-2021\\Machine " +
            "Learning\\Week 2 - Decision Trees\\Week 2 Live " +
            "Class\\tsml-master\\src\\main\\java\\experiments\\data\\uci\\iris\\";


    public static void measuringIG(){
        //Play golf
        double[][] t= {{2,2},{7,3}};
        double[][] o= {{2,3},{4,0},{3,2}};
        double[][] h=  {{5,4},{4,1}};
        double[][] w=  {{6,2},{3,3}};
        DecimalFormat df = new DecimalFormat("##.#####");
        InfoGainSplitCrit infoGain = new InfoGainSplitCrit();
        Distribution dist= new Distribution(o);
        double outlookIG = infoGain.splitCritValue(dist);
        System.out.println(" Outlook IG = "+df.format(1/outlookIG));

        dist= new Distribution(t);
        double tempIG=infoGain.splitCritValue(dist);
        System.out.println(" Temp IG = "+df.format(1/tempIG));
        dist= new Distribution(h);
        double humIG=infoGain.splitCritValue(dist);
        System.out.println(" Humidity IG = "+df.format(1/humIG));
        dist= new Distribution(w);
        double windyIG=infoGain.splitCritValue(dist);
        System.out.println(" Windy IG = "+df.format(1/windyIG));






    }
    public static void main(String[] args) throws Exception {
        measuringIG();
        System.exit(0);

        Instances all=experiments.data.DatasetLoading.loadData(dataPath+"iris");
        J48 c45 = new J48();
        System.out.println("Capabilities = "+c45.getCapabilities());
        c45.buildClassifier(all);
        System.out.println(c45);

        System.out.println(" Does it reduce error prune by default? "+c45.getReducedErrorPruning());
        c45.setUnpruned(true);
        c45.buildClassifier(all);
        System.out.println(c45);


        c45.setReducedErrorPruning(true);
        c45.buildClassifier(all);
        System.out.println(c45);



    }

}
