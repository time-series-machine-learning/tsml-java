package ml6002b2022;

import experiments.data.DatasetLoading;
import weka.core.Instances;

import java.io.FileReader;
import java.io.IOException;

public class Week1Examples {
    public static void main(String[] args) throws IOException {
        Instances wdbc = DatasetLoading.loadData("src/main/java/ml6002b2022/wdbc");
        FileReader reader = new FileReader("src/main/java/experiments/data/uci/iris/iris.arff");
        Instances iris = new Instances(reader);
        System.out.println(" Number classes WDBC= "+wdbc.numClasses());
        System.out.println(" Number classes iris= "+wdbc.numClasses());

    }


}
