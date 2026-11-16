"""references.py -- the reference list (IEEE style). Bibliographic details were checked against the publishers' records
(Crossref / DOI landing pages, CVF, NeurIPS, PMLR) on 16 Nov. 2026.  *...* = italics."""

REFS = {
    "garrido": "S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez, “Automatic generation and detection of highly "
               "reliable fiducial markers under occlusion,” *Pattern Recognit.*, vol. 47, no. 6, pp. 2280–2292, Jun. 2014, doi: 10.1016/j.patcog.2014.01.005.",
    "olson": "E. Olson, “AprilTag: A robust and flexible visual fiducial system,” in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, Shanghai, China, "
             "May 2011, pp. 3400–3407, doi: 10.1109/ICRA.2011.5979561.",
    "romero": "F. J. Romero-Ramirez, R. Muñoz-Salinas, and R. Medina-Carnicer, “Speeded up detection of squared fiducial markers,” *Image Vis. "
              "Comput.*, vol. 76, pp. 38–47, Aug. 2018, doi: 10.1016/j.imavis.2018.05.004.",
    "kendall": "A. Kendall, M. Grimes, and R. Cipolla, “PoseNet: A convolutional network for real-time 6-DOF camera relocalization,” in *Proc. IEEE "
               "Int. Conf. Comput. Vis. (ICCV)*, Santiago, Chile, Dec. 2015, pp. 2938–2946, doi: 10.1109/ICCV.2015.336.",
    "xiang": "Y. Xiang, T. Schmidt, V. Narayanan, and D. Fox, “PoseCNN: A convolutional neural network for 6D object pose estimation in cluttered "
             "scenes,” in *Proc. Robot.: Sci. Syst. (RSS)*, Pittsburgh, PA, USA, Jun. 2018, doi: 10.15607/RSS.2018.XIV.019.",
    "collins": "T. Collins and A. Bartoli, “Infinitesimal plane-based pose estimation,” *Int. J. Comput. Vis.*, vol. 109, no. 3, pp. 252–286, "
               "Sep. 2014, doi: 10.1007/s11263-014-0725-5.",
    "schweighofer": "G. Schweighofer and A. Pinz, “Robust pose estimation from a planar target,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 28, "
                    "no. 12, pp. 2024–2030, Dec. 2006, doi: 10.1109/TPAMI.2006.252.",
    "he": "K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. "
          "(CVPR)*, Las Vegas, NV, USA, Jun. 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.",
    "zhou": "Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, “On the continuity of rotation representations in neural networks,” in *Proc. IEEE/CVF "
            "Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Long Beach, CA, USA, Jun. 2019, pp. 5738–5746, doi: 10.1109/CVPR.2019.00589.",
    "li_cdpn": "Z. Li, G. Wang, and X. Ji, “CDPN: Coordinates-based disentangled pose network for real-time RGB-based 6-DoF object pose "
               "estimation,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Seoul, South Korea, Oct. 2019, pp. 7677–7686, doi: 10.1109/ICCV.2019.00777.",
    "liu_coordconv": "R. Liu *et al.*, “An intriguing failing of convolutional neural networks and the CoordConv solution,” in *Adv. Neural Inf. "
                     "Process. Syst. (NeurIPS)*, vol. 31, Montréal, QC, Canada, Dec. 2018, pp. 9605–9616.",
    "tobin": "J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, “Domain randomization for transferring deep neural networks from "
             "simulation to the real world,” in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Vancouver, BC, Canada, Sep. 2017, pp. 23–30, "
             "doi: 10.1109/IROS.2017.8202133.",
    "hu_charuco": "D. Hu, D. DeTone, and T. Malisiewicz, “Deep ChArUco: Dark ChArUco marker pose estimation,” in *Proc. IEEE/CVF Conf. Comput. Vis. "
                  "Pattern Recognit. (CVPR)*, Long Beach, CA, USA, Jun. 2019, pp. 8428–8436, doi: 10.1109/CVPR.2019.00863.",
    "tremblay": "J. Tremblay, T. To, B. Sundaralingam, Y. Xiang, D. Fox, and S. Birchfield, “Deep object pose estimation for semantic robotic "
                "grasping of household objects,” in *Proc. 2nd Conf. Robot Learn. (CoRL)*, Proc. Mach. Learn. Res., vol. 87, Oct. 2018, pp. 306–316.",
    "corke": "P. Corke, *Robotics, Vision and Control: Fundamental Algorithms in Python*, 3rd ed. (Springer Tracts in Advanced Robotics, vol. 146). "
             "Cham, Switzerland: Springer, 2023, doi: 10.1007/978-3-031-06469-2.",
    "lectures": "*Machine Vision*, Lectures 4–5 and starter code (mvlib.py), course material, “Gheorghe Asachi” Technical University of Iași, "
                "Iași, Romania, 2026 (unpublished).",
    "brown": "D. C. Brown, “Decentering distortion of lenses,” *Photogramm. Eng.*, vol. 32, no. 3, pp. 444–462, 1966.",
    "zhang": "Z. Zhang, “A flexible new technique for camera calibration,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 22, no. 11, "
             "pp. 1330–1334, Nov. 2000, doi: 10.1109/34.888718.",
    "levenberg": "K. Levenberg, “A method for the solution of certain non-linear problems in least squares,” *Quart. Appl. Math.*, vol. 2, no. 2, "
                 "pp. 164–168, Jul. 1944, doi: 10.1090/qam/10666.",
    "marquardt": "D. W. Marquardt, “An algorithm for least-squares estimation of nonlinear parameters,” *J. Soc. Ind. Appl. Math.*, vol. 11, no. 2, "
                 "pp. 431–441, Jun. 1963, doi: 10.1137/0111030.",
    "huynh": "D. Q. Huynh, “Metrics for 3D rotations: Comparison and analysis,” *J. Math. Imaging Vis.*, vol. 35, no. 2, pp. 155–164, Oct. 2009, "
             "doi: 10.1007/s10851-009-0161-2.",
    "kalman": "R. E. Kalman, “A new approach to linear filtering and prediction problems,” *J. Basic Eng.*, vol. 82, no. 1, pp. 35–45, Mar. 1960, "
              "doi: 10.1115/1.3662552.",
    "barshalom": "Y. Bar-Shalom, X.-R. Li, and T. Kirubarajan, *Estimation with Applications to Tracking and Navigation: Theory, Algorithms and "
                 "Software*. New York, NY, USA: Wiley, 2001, doi: 10.1002/0471221279.",
    "bradski": "G. Bradski, “The OpenCV library,” *Dr. Dobb’s J. Softw. Tools*, Nov. 2000.",
    "opencv": "OpenCV, “OpenCV 5.0 documentation: Camera calibration; 3D vision functionality; ArUco markers and boards detection for robust "
              "camera pose estimation.” [Online]. Available: https://docs.opencv.org/5.0/ (accessed Nov. 16, 2026).",
    "paszke": "A. Paszke *et al.*, “PyTorch: An imperative style, high-performance deep learning library,” in *Adv. Neural Inf. Process. Syst. "
              "(NeurIPS)*, vol. 32, Vancouver, BC, Canada, Dec. 2019, pp. 8026–8037.",
    "deng": "J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in *Proc. IEEE "
            "Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Miami, FL, USA, Jun. 2009, pp. 248–255, doi: 10.1109/CVPR.2009.5206848.",
    "loshchilov": "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in *Proc. 7th Int. Conf. Learn. Represent. (ICLR)*, "
                  "New Orleans, LA, USA, May 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7.",
    "smith": "L. N. Smith and N. Topin, “Super-convergence: Very fast training of neural networks using large learning rates,” in *Proc. SPIE*, "
             "vol. 11006, Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications, Baltimore, MD, USA, May 2019, "
             "Art. no. 1100612, doi: 10.1117/12.2520589.",
    "moakher": "M. Moakher, “Means and averaging in the group of rotations,” *SIAM J. Matrix Anal. Appl.*, vol. 24, no. 1, pp. 1–16, Jan. 2002, "
               "doi: 10.1137/S0895479801383877.",
}
