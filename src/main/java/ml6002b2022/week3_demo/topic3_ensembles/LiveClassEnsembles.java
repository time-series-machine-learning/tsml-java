package ml6002b2022.week3_demo.topic3_ensembles;

import utilities.InstanceTools;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class LiveClassEnsembles {
    static String dataPath="C:\\Users\\Tony\\OneDrive - University of East Anglia\\Teaching\\2020-2021\\Machine " +
            "Learning\\Week 2 - Decision Trees\\Week 2 Live " +
            "Class\\tsml-master\\src\\main\\java\\experiments\\data\\uci\\iris\\";


    public static void main(String[] args) throws Exception {
        Instances all=experiments.data.DatasetLoading.loadData(dataPath+"iris");
        Instances[] split= InstanceTools.resampleInstances(all,0,0.5);


        MyEnsemble te= new MyEnsemble();
        J48 c45 = new J48();
        te.buildClassifier(split[0]);
        int count=0;
        c45.buildClassifier(split[0]);
        int count2=0;
        for(Instance ins:split[1]){
            int pred = (int)te.classifyInstance(ins);
            int pred2 = (int)c45.classifyInstance(ins);
            int actual=(int)ins.classValue();
//            System.out.println(" pred = "+pred+" actual = "+actual);
            if(pred==actual)
                count++;
            if(pred2==actual)
                count2++;

        }
        System.out.println(" TEST accuracy = "+count/(double)all.numInstances());
        System.out.println(" TEST accuracy = "+count2/(double)all.numInstances());

    }


}
