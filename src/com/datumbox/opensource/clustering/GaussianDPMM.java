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
package com.datumbox.opensource.clustering;

import com.datumbox.opensource.dataobjects.Point;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 */
public class GaussianDPMM extends DPMM<GaussianDPMM.Cluster>{

    /**
     * Multivariate Normal with Normal-Inverse-Wishart prior.
     * References:
     *      http://snippyhollow.github.io/blog/2013/03/10/collapsed-gibbs-sampling-for-dirichlet-process-gaussian-mixture-models/
     *      http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process/
     *      http://www.cs.princeton.edu/courses/archive/fall07/cos597C/scribe/20070921.pdf
     * 
     * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
     */
    public class Cluster extends DPMM.Cluster {
        

        //hyper parameters
        private final Integer kappa0;
        private final Integer nu0;
        private final RealVector mu0;
        private final RealMatrix psi0;
        
        //cluster parameters
        private RealVector mean;
        private RealMatrix covariance;
        
        //validation - confidence interval vars
        private RealMatrix meanError;
        private int meanDf;
        
        /**
         * Sum of observations used in calculation of cluster clusterParameters such 
         * as mean.
         */
        private RealVector xi_sum;
        /**
         * Sum of squared of observations used in calculation of cluster 
         * clusterParameters such as variance.
         */
        private RealMatrix xi_square_sum;

        /**
         * Cached value of Covariance determinant used only for speed optimization
         */
        private Double cache_covariance_determinant;

        /**
         * Cached value of Inverse Covariance used only for speed optimization
         */
        private RealMatrix cache_covariance_inverse;

        /**
         * Constructor of Gaussian Mixtrure Cluster
         * 
         * @param dimensionality    The dimensionality of the data that we cluster
         * @param kappa0    Mean fraction parameter
         * @param nu0   Degrees of freedom for Inverse-Wishart
         * @param mu0   Mean vector for Normal
         * @param psi0  Pairwise deviation product of Inverse Wishart
         */
        public Cluster(Integer dimensionality, Integer kappa0, Integer nu0, RealVector mu0, RealMatrix psi0) {
            super(dimensionality);
            
            //Set default hyperparameters if not set
            if(kappa0==null) {
                kappa0 = 0;
            }

            if(nu0==null || nu0<dimensionality) {
                nu0 = dimensionality;
            }

            if(mu0==null) {
                mu0 = new ArrayRealVector(dimensionality); //0 vector
            }

            if(psi0==null) {
                psi0 = MatrixUtils.createRealIdentityMatrix(dimensionality); //identity matrix
            }
            
            this.kappa0 = kappa0;
            this.nu0 = nu0;
            this.mu0 = mu0;
            this.psi0 = psi0;
            
            mean = new ArrayRealVector(dimensionality);
            covariance = MatrixUtils.createRealIdentityMatrix(dimensionality);

            meanError = calculateMeanError(psi0, kappa0, nu0);
            meanDf = Math.max(0, nu0-dimensionality+1);
        }
        
        /**
         * Adds a single point in the cluster.
         * 
         * @param xi    The point that we wish to add in the cluster.
         */
        @Override
        public void addPoint(Point xi) {
            addSinglePoint(xi);
            updateClusterParameters();
        }

        /**
         * Internal method that adds the point int cluster and updates clusterParams
         * 
         * @param xi    The point that we wish to add in the cluster.
         */
        private void addSinglePoint(Point xi) {
            int nk= pointList.size();

            //update cluster clusterParameters
            if(nk==0) {
                xi_sum=xi.data;
                xi_square_sum=xi.data.outerProduct(xi.data);
            }
            else {
                xi_sum=xi_sum.add(xi.data);
                xi_square_sum=xi_square_sum.add(xi.data.outerProduct(xi.data));
            }

            pointList.add(xi);
        }

        /**
         * Removes a point from the cluster.
         * 
         * @param xi    The point that we wish to remove from the cluster
         */
        @Override
        public void removePoint(Point xi) {
            int index = pointList.indexOf(xi);
            if(index==-1) {
                return;
            }

            //update cluster clusterParameters
            xi_sum=xi_sum.subtract(xi.data);
            xi_square_sum=xi_square_sum.subtract(xi.data.outerProduct(xi.data));

            pointList.remove(index);

            updateClusterParameters();
        }

        private RealMatrix calculateMeanError(RealMatrix Psi, int kappa, int nu) {
            //Reference: page 18, equation 228 at http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
            return Psi.scalarMultiply(1.0/(kappa*(nu-dimensionality+1.0)));
        }

        /**
         * Updates the cluster's internal parameters based on the stored information.
         */
        @Override
        protected void updateClusterParameters() {
            int n = pointList.size();
            if(n<=0) {
                return;
            }

            int kappa_n = kappa0 + n;
            int nu = nu0 + n;

            RealVector mu = xi_sum.mapDivide(n);
            RealVector mu_mu_0 = mu.subtract(mu0);

            RealMatrix C = xi_square_sum.subtract( ( mu.outerProduct(mu) ).scalarMultiply(n) );

            RealMatrix psi = psi0.add( C.add( ( mu_mu_0.outerProduct(mu_mu_0) ).scalarMultiply(kappa0*n/(double)kappa_n) ));
            C = null;
            mu_mu_0 = null;

            mean = ( mu0.mapMultiply(kappa0) ).add( mu.mapMultiply(n) ).mapDivide(kappa_n);
            covariance = psi.scalarMultiply(  (kappa_n+1.0)/(kappa_n*(nu - dimensionality + 1.0))  );

            //clear cache
            cache_covariance_determinant=null;
            cache_covariance_inverse=null;

            meanError = calculateMeanError(psi, kappa_n, nu);
            meanDf = Math.max(0, nu-dimensionality+1);
        }

        /**
         * Returns the log posterior PDF of a particular point xi, to belong to this
         * cluster.
         * 
         * @param xi    The point for which we want to estimate the PDF.
         * @return      The log posterior PDF
         */
        @Override
        public double posteriorLogPdf(Point xi) {
            RealVector x_mu = xi.data.subtract(mean);

            if(cache_covariance_determinant==null || cache_covariance_inverse==null) {
                LUDecomposition lud = new LUDecomposition(covariance);
                cache_covariance_determinant = lud.getDeterminant();
                cache_covariance_inverse = lud.getSolver().getInverse();
                lud =null;
            }
            Double determinant=cache_covariance_determinant;
            RealMatrix invCovariance=cache_covariance_inverse;

            double x_muInvSx_muT = (invCovariance.preMultiply(x_mu)).dotProduct(x_mu);

            double normConst = 1.0/( Math.pow(2*Math.PI, dimensionality/2.0) * Math.pow(determinant, 0.5) );


            //double pdf = Math.exp(-0.5 * x_muInvSx_muT)*normConst;
            double logPdf = -0.5 * x_muInvSx_muT + Math.log(normConst);
            return logPdf;
        }
        
        /**
         * Getter for the mean of the cluster.
         * 
         * @return  The mean vector
         */
        public RealVector getMean() {
            return mean;
        }
        
        /**
         * Getter for the covariance of the cluster.
         * 
         * @return  The Covariance Matrix
         */
        public RealMatrix getCovariance() {
            return covariance;
        }

        /**
         * Getter for Mean Error of the cluster.
         * 
         * @return  The Mean Error Matrix
         */
        public RealMatrix getMeanError() {
            return meanError;
        }
        
        /**
         * The degrees of freedom of Student's Distribution.
         * http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf Equation 228
         * 
         * @return 
         */
        public int getMeanDf() {
            return meanDf;
        }

    }

   
    private final Integer kappa0;
    private final Integer nu0;
    private final RealVector mu0;
    private final RealMatrix psi0;
    
    /**
     * Constructor of Gaussian DPMM.
     * 
     * @param dimensionality    The dimensionality of the data that we cluster
     * @param alpha     The alpha of the Dirichlet Process
     * @param kappa0    Mean fraction parameter
     * @param nu0   Degrees of freedom for Inverse-Wishart
     * @param mu0   Mean vector for Normal
     * @param psi0  Pairwise deviation product of Inverse Wishart
     */
    public GaussianDPMM(Integer dimensionality, Double alpha, Integer kappa0, Integer nu0, RealVector mu0, RealMatrix psi0) {
        super(dimensionality, alpha);
            
        this.kappa0 = kappa0;
        this.nu0 = nu0;
        this.mu0 = mu0;
        this.psi0 = psi0;
    }

    @Override
    protected Cluster generateCluster() {
        return new GaussianDPMM.Cluster(dimensionality, kappa0, nu0, mu0, psi0);        
    }
}
