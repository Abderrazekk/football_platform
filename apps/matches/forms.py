from django import forms

from apps.matches.models import Team


class PredictionForm(forms.Form):
    home_team = forms.ModelChoiceField(
        queryset=Team.objects.none(),
        label="Équipe à domicile",
        empty_label="— Choisir une équipe —",
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    away_team = forms.ModelChoiceField(
        queryset=Team.objects.none(),
        label="Équipe à l'extérieur",
        empty_label="— Choisir une équipe —",
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["home_team"].queryset = Team.objects.order_by("name")
        self.fields["away_team"].queryset = Team.objects.order_by("name")

    def clean(self):
        cleaned = super().clean()
        home = cleaned.get("home_team")
        away = cleaned.get("away_team")
        if home and away and home == away:
            raise forms.ValidationError("Les deux équipes doivent être différentes.")
        return cleaned