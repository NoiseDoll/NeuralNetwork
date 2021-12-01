using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralUI.Framework;

namespace NeuralUI.ViewModels
{
    public class ConnectionViewModel : ViewModelBase
    {
        private float weight;

        private double x1;
        private double x2;
        private double y1;
        private double y2;

        private double textWidth;
        private double textHeight;

        public float Weight
        {
            get => weight;
            set => SetProperty(ref weight, value);
        }

        public double X1
        {
            get => x1;
            set => SetProperty(ref x1, value, () => NotifyPropertyChanged(nameof(TextLeft)));
        }

        public double X2
        {
            get => x2;
            set => SetProperty(ref x2, value, () => NotifyPropertyChanged(nameof(TextLeft)));
        }

        public double Y1
        {
            get => y1;
            set => SetProperty(ref y1, value, () => NotifyPropertyChanged(nameof(TextTop)));
        }

        public double Y2
        {
            get => y2;
            set => SetProperty(ref y2, value, () => NotifyPropertyChanged(nameof(TextTop)));
        }

        public double TextWidth
        {
            get => textWidth;
            set => SetProperty(ref textWidth, value, () => NotifyPropertyChanged(nameof(TextLeft)));
        }

        public double TextHeight
        {
            get => textHeight;
            set => SetProperty(ref textHeight, value, () => NotifyPropertyChanged(nameof(TextTop)));
        }

        public double TextLeft
        {
            get
            {
                var x = (X2 - X1) * 0.5;
                x -= TextWidth * 0.5;
                return x;
            }
        }

        public double TextTop
        {
            get
            {
                var y = (Y2 - Y1) * 0.5;
                y -= TextHeight * 0.5;
                y += 5;
                return y;
            }
        }

    }
}
