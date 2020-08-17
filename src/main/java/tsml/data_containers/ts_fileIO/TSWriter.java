package tsml.data_containers.ts_fileIO;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.io.*;
import java.text.DecimalFormat;
import java.util.Arrays;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

public class TSWriter {

    TimeSeriesInstances data;
    BufferedWriter writer;

    public void setData(TimeSeriesInstances dat){
        data = dat;
    }

    DecimalFormat df = new DecimalFormat();

    public void setDestination(OutputStream output) {
        writer = new BufferedWriter(new OutputStreamWriter(output));
    }

    public void setDesination(File output) throws FileNotFoundException {
        setDestination(new FileOutputStream(output));
    }

    BufferedWriter getWriter(){
        return writer;
    }

    
    public TSWriter(){
        df.setMaximumFractionDigits(6);
    }

    public TSWriter(File output) throws IOException{
        this();
        setDesination(output);
    }

    public void writeBatch(){

        PrintWriter outW = new PrintWriter(getWriter());

        //writer header info first.
        outW.println("@problemName " + data.getProblemName());
        outW.println("@timeStamps " + data.hasTimeStamps());
        outW.println("@missing " + data.hasMissing());
        outW.println("@univariate " + !data.isMultivariate());
        outW.println("@dimensions " + data.getMaxNumChannels());
        outW.println("@equalLength " + data.isEqualLength());
        outW.println("@seriesLength " + data.getMaxLength());
        //outW.println("@classLabel " + );
        outW.print("@classLabel ");
        outW.print(data.getClassLabels() != null && data.getClassLabels().length > 0);
        outW.println(data.getClassLabelsFormatted());

        outW.println("@data");
        //then writer data.
        StringBuilder sb = new StringBuilder();
        for(TimeSeriesInstance inst : data){
            for(TimeSeries ts : inst){
                
                for(Double d : ts.getSeries())
                    sb.append(df.format(d.doubleValue())).append(",");
                sb.replace(sb.length()-1,sb.length(),":"); //we use colon to separate dimensions, overwriter the last comma.
            }
            sb.append(data.getClassLabels()[inst.getLabelIndex()]); //append the class label.
            sb.append("\n");
        }

        outW.print(sb.toString());
        outW.close();
    }
}