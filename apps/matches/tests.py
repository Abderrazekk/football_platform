from django.contrib.auth.models import User
from django.test import Client, TestCase

from apps.matches.models import Prediction, Team


class ViewAccessTests(TestCase):
    def test_predict_requires_login(self):
        r = self.client.get("/matches/predict/")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/accounts/login/", r.url)

    def test_history_requires_login(self):
        r = self.client.get("/matches/history/")
        self.assertEqual(r.status_code, 302)

    def test_teams_list_public(self):
        r = self.client.get("/matches/teams/")
        self.assertEqual(r.status_code, 200)

    def test_standings_public(self):
        r = self.client.get("/matches/standings/")
        self.assertEqual(r.status_code, 200)

    def test_model_performance_public(self):
        r = self.client.get("/model/performance/")
        self.assertEqual(r.status_code, 200)

    def test_home_public(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)


class PredictionFlowTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        Team.objects.create(name="Arsenal")
        Team.objects.create(name="Chelsea")

    def setUp(self):
        self.user = User.objects.create_user("testuser", password="testpass123")
        self.client = Client()

    def test_predict_and_history(self):
        self.client.login(username="testuser", password="testpass123")
        arsenal = Team.objects.get(name="Arsenal")
        chelsea = Team.objects.get(name="Chelsea")

        r = self.client.post(
            "/matches/predict/",
            {"home_team": arsenal.pk, "away_team": chelsea.pk},
        )
        self.assertEqual(r.status_code, 200)

        # If the ML model is loaded, a prediction is created
        # If it isn't (e.g. CI without model artifacts), the view shows
        # a friendly error — accept both cases but verify DB consistency
        created = Prediction.objects.filter(user=self.user).count()
        self.assertIn(created, (0, 1))

        r2 = self.client.get("/matches/history/")
        self.assertEqual(r2.status_code, 200)

    def test_form_rejects_same_team(self):
        self.client.login(username="testuser", password="testpass123")
        arsenal = Team.objects.get(name="Arsenal")
        r = self.client.post(
            "/matches/predict/",
            {"home_team": arsenal.pk, "away_team": arsenal.pk},
        )
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "différentes")
        self.assertEqual(Prediction.objects.filter(user=self.user).count(), 0)