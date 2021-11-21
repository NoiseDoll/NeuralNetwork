using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using NeuralNetwork;
using NeuralUI.Framework;

namespace NeuralUI.ViewModels
{
    public class MainViewModel : ViewModelBase
    {
        private Perceptron perceptron;

        private NeuralNetworkViewModel nn;

        public NeuralNetworkViewModel NN
        {
            get => nn;
            set => SetProperty(ref nn, value);
        }

        public ICommand CreateNNCommand { get; }

        public MainViewModel()
        {
            CreateNNCommand = new RelayCommand(CreateNN);
        }

        private async void CreateNN(object obj)
        {
            perceptron = Perceptron.CreateRandom(4, 1, 2, 12, new Random());
            NN = new NeuralNetworkViewModel(perceptron);
        }
    }
}
