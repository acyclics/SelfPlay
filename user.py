from sp_core import SelfPlay

if __name__ == "__main__":
    game = SelfPlay()
    game.process_interface()

'''
[[<tf.Variable 'oldpi/pi_l1/kernel:0' shape=(3, 400) dtype=float32_ref>, <tf.Variable 'oldpi/pi_l1/bias:0' shape=(400,) dtype=float32_ref>, <tf.Variable 'oldpi/pi_l2/kernel:0' shape=(400, 400) dtype=float32_ref>, <tf.Variable 'oldpi/pi_l2/bias:0' shape=(400,) dtype=float32_ref>, <tf.Variable 'oldpi/pi_mu/kernel:0' shape=(400, 1) dtype=float32_ref>, <tf.Variable 'oldpi/pi_mu/bias:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'oldpi/pi_sigma:0' shape=(1,) dtype=float32_ref>], [<tf.Variable 'pi/pi_l1/kernel:0' shape=(3, 400) dtype=float32_ref>, <tf.Variable 'pi/pi_l1/bias:0' shape=(400,) dtype=float32_ref>, <tf.Variable 'pi/pi_l2/kernel:0' shape=(400, 400) dtype=float32_ref>, <tf.Variable 'pi/pi_l2/bias:0' shape=(400,) dtype=float32_ref>, <tf.Variable 'pi/pi_mu/kernel:0' shape=(400, 1) dtype=float32_ref>, <tf.Variable 'pi/pi_mu/bias:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'pi/pi_sigma:0' shape=(1,) dtype=float32_ref>], [<tf.Variable
'oldvf/vf_l1/kernel:0' shape=(3, 400) dtype=float32_ref>, <tf.Variable 'oldvf/vf_l1/bias:0' shape=(400,) dtype=float32_ref>, <tf.Variable 'oldvf/vf_l2/kernel:0' shape=(400, 400) dtype=float32_ref>, <tf.Variable 'oldvf/vf_l2/bias:0' shape=(400,) dtype=float32_ref>, <tf.Variable 'oldvf/vf_output/kernel:0' shape=(400, 1) dtype=float32_ref>, <tf.Variable 'oldvf/vf_output/bias:0' shape=(1,) dtype=float32_ref>], [<tf.Variable 'vf/vf_l1/kernel:0' shape=(3, 400) dtype=float32_ref>, <tf.Variable 'vf/vf_l1/bias:0' shape=(400,) dtype=float32_ref>, <tf.Variable 'vf/vf_l2/kernel:0' shape=(400, 400) dtype=float32_ref>, <tf.Variable 'vf/vf_l2/bias:0' shape=(400,) dtype=float32_ref>, <tf.Variable 'vf/vf_output/kernel:0' shape=(400, 1) dtype=float32_ref>, <tf.Variable 'vf/vf_output/bias:0' shape=(1,) dtype=float32_ref>]]
'''
