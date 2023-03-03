import os 
import numpy as np
import subprocess

# def ratio(losses, thresholds=[1,2,3,4,5,6,7,8,9,10]):
def ratio(losses, thresholds=[5,10]):
    return [
        '{:.3f}'.format(np.mean(losses < threshold))
        for threshold in thresholds
    ]

if __name__ == '__main__':
    scene = 'Indoor'
    dir_base = 'result_errors/Indoor/'
    save_pt = 'resultfinal_errors/Indoor/'

    subprocess.check_output(['mkdir', '-p', save_pt])

    with open(save_pt +'ratio_methods_'+scene+'.txt','w') as f:
        f.write('5deg 10deg'+'\n')
        pair_list = os.listdir(dir_base)
        enhancer = os.listdir(dir_base+'/pair9/')
        for method in enhancer:
            pose_error_list = sorted(os.listdir(dir_base+'/pair9/'+method))
            for pose_error in pose_error_list:
                error_array = np.expand_dims(np.zeros((6, 8)),axis=2)
                for pair in pair_list:
                    try:
                        error = np.expand_dims(np.load(dir_base+'/'+pair+'/'+method+'/'+pose_error),axis=2)
                    except:
                        print('error in', dir_base+'/'+pair+'/'+method+'/'+pose_error)
                        continue
                    error_array = np.concatenate((error_array,error),axis=2)
                ratio_result = ratio(error_array[:,:,1::].flatten())
                f.write(method + '_' + pose_error[11:-4] +' '+' '.join([str(i) for i in ratio_result])+"\n")

    
    scene = 'Outdoor'
    dir_base = 'result_errors/Outdoor/'
    save_pt = 'resultfinal_errors/Outdoor/'

    subprocess.check_output(['mkdir', '-p', save_pt])

    with open(save_pt +'ratio_methods_'+scene+'.txt','w') as f:
        f.write('5deg 10deg'+'\n')
        pair_list = os.listdir(dir_base)
        enhancer = os.listdir(dir_base+'/pair9/')
        for method in enhancer:
            pose_error_list = sorted(os.listdir(dir_base+'/pair9/'+method))
            for pose_error in pose_error_list:
                error_array = np.expand_dims(np.zeros((6, 8)),axis=2)
                for pair in pair_list:
                    error = np.expand_dims(np.load(dir_base+'/'+pair+'/'+method+'/'+pose_error),axis=2)
                    error_array = np.concatenate((error_array,error),axis=2)
                ratio_result = ratio(error_array[:,:,1::].flatten())
                f.write(method + '_' + pose_error[11:-4] +' '+' '.join([str(i) for i in ratio_result])+"\n")
