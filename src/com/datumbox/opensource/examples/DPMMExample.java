/* 
 * Copyright (C) 2014 Vasilis Vryniotis <bbriniotis at datumbox.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.datumbox.opensource.examples;

import com.datumbox.opensource.dataobjects.Point;
import com.datumbox.opensource.clustering.DPMM;
import com.datumbox.opensource.clustering.GaussianDPMM;
import com.datumbox.opensource.clustering.MultinomialDPMM;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;


/**
 * Demos of Dirichlet Process Mixture Model.
 * 
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 */
public class DPMMExample {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        GDPMM();
        System.out.println();
        MDPMM();
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        System.out.println("Completed in "+String.valueOf(elapsedTime/1000.0)+" sec");
    }
    
    /**
     * Demo of Dirichlet Process Mixture Model with Multinomial
     */
    public static void MDPMM() {
        System.out.println("Multinomial DPMM");
        
        //Data points to cluster
        List<Point> pointList = new ArrayList<>();
        //cluster 1
        pointList.add(new Point(0, new ArrayRealVector(new double[]{10.0,13.0, 5.0,6.0,5.0,4.0, 0.0,0.0,0.0,0.0})));
        pointList.add(new Point(1, new ArrayRealVector(new double[]{11.0,11.0, 6.0,7.0,7.0,3.0, 0.0,0.0,1.0,0.0})));
        pointList.add(new Point(2, new ArrayRealVector(new double[]{12.0,12.0, 10.0,16.0,4.0,6.0, 0.0,0.0,0.0,2.0})));
        //cluster 2
        pointList.add(new Point(3, new ArrayRealVector(new double[]{10.0,13.0, 0.0,0.0,0.0,0.0, 5.0,6.0,5.0,4.0})));
        pointList.add(new Point(4, new ArrayRealVector(new double[]{11.0,11.0, 0.0,0.0,1.0,0.0, 6.0,7.0,7.0,3.0})));
        pointList.add(new Point(5, new ArrayRealVector(new double[]{12.0,12.0, 0.0,0.0,0.0,2.0, 10.0,16.0,4.0,6.0})));
        //cluster 3
        pointList.add(new Point(6, new ArrayRealVector(new double[]{10.0,13.0, 5.0,6.0,5.0,4.0, 5.0,6.0,5.0,4.0})));
        pointList.add(new Point(7, new ArrayRealVector(new double[]{11.0,11.0, 6.0,7.0,7.0,3.0, 6.0,7.0,7.0,3.0})));
        pointList.add(new Point(8, new ArrayRealVector(new double[]{12.0,12.0, 10.0,16.0,4.0,6.0, 10.0,16.0,4.0,6.0})));
        
        
        //Dirichlet Process parameter
        Integer dimensionality = 10;
        double alpha = 1.0;
        
        //Hyper parameters of Base Function
        double alphaWords = 1.0;
        
        //Create a DPMM object
        DPMM dpmm = new MultinomialDPMM(dimensionality, alpha, alphaWords);
        
        int maxIterations = 100;
        int performedIterations = dpmm.cluster(pointList, maxIterations);
        if(performedIterations<maxIterations) {
            System.out.println("Converged in "+String.valueOf(performedIterations));
        }
        else {
            System.out.println("Max iterations of "+String.valueOf(performedIterations)+" reached. Possibly did not converge.");
        }
        
        //get a list with the point ids and their assignments
        Map<Integer, Integer> zi = dpmm.getPointAssignments();
        System.out.println(zi.toString());
        
    }
    
    /**
     * Demo of Dirichlet Process Mixture Model with Gaussian
     */
    public static void GDPMM() {
        System.out.println("Gaussian DPMM");
        
        //Data points to cluster
        List<Point> pointList = new ArrayList<>();
        //cluster 1
        pointList.add(new Point(0, new ArrayRealVector(new double[]{5.0,1.0})));
        pointList.add(new Point(1, new ArrayRealVector(new double[]{5.1,1.1})));
        pointList.add(new Point(2, new ArrayRealVector(new double[]{4.9,0.9})));
        //cluster 2
        pointList.add(new Point(3, new ArrayRealVector(new double[]{15.0,11.0})));
        pointList.add(new Point(4, new ArrayRealVector(new double[]{15.1,11.1})));
        pointList.add(new Point(5, new ArrayRealVector(new double[]{14.9,10.9})));
        //cluster 3
        pointList.add(new Point(6, new ArrayRealVector(new double[]{1.0,5.0})));
        pointList.add(new Point(7, new ArrayRealVector(new double[]{1.1,5.1})));
        pointList.add(new Point(8, new ArrayRealVector(new double[]{0.9,4.9})));
        
        //Dirichlet Process parameter
        Integer dimensionality = 2;
        double alpha = 1.0;
        
        //Hyper parameters of Base Function
        int kappa0 = 0;
        int nu0 = 1;
        RealVector mu0 = new ArrayRealVector(new double[]{0.0, 0.0});
        RealMatrix psi0 = new BlockRealMatrix(new double[][]{{1.0,0.0},{0.0,1.0}});
        
        //Create a DPMM object
        DPMM dpmm = new GaussianDPMM(dimensionality, alpha, kappa0, nu0, mu0, psi0);
        
        int maxIterations = 100;
        int performedIterations = dpmm.cluster(pointList, maxIterations);
        if(performedIterations<maxIterations) {
            System.out.println("Converged in "+String.valueOf(performedIterations));
        }
        else {
            System.out.println("Max iterations of "+String.valueOf(performedIterations)+" reached. Possibly did not converge.");
        }
        
        //get a list with the point ids and their assignments
        Map<Integer, Integer> zi = dpmm.getPointAssignments();
        System.out.println(zi.toString());
        
    }
}
