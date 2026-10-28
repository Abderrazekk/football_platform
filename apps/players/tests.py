from django.contrib.auth.models import User
from django.test import Client, TestCase

from apps.matches.models import Season, Team
from apps.players.models import Player, PlayerSeasonStats


class PlayersPublicAccessTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        Team.objects.create(name="Arsenal")
        Player.objects.create(
            element_code=999999, name="Test Player",
            position="MID",
        )

    def test_players_list_public(self):
        r = self.client.get("/players/")
        self.assertEqual(r.status_code, 200)

    def test_players_list_search(self):
        r = self.client.get("/players/?q=Test")
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Test Player")

    def test_player_detail_public(self):
        p = Player.objects.get(element_code=999999)
        r = self.client.get(f"/players/{p.pk}/")
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Test Player")

    def test_player_detail_404(self):
        r = self.client.get("/players/999999/")
        self.assertEqual(r.status_code, 404)


class PlayersRestrictedTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        Team.objects.create(name="Arsenal")
        Team.objects.create(name="Chelsea")
        cls.user = User.objects.create_user("tester", password="testpass123")

    def test_compare_requires_login(self):
        r = self.client.get("/players/compare/")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/accounts/login/", r.url)

    def test_match_stars_requires_login(self):
        r = self.client.get("/players/match-stars/")
        self.assertEqual(r.status_code, 302)

    def test_at_risk_requires_login(self):
        r = self.client.get("/players/at-risk/")
        self.assertEqual(r.status_code, 302)

    def test_compare_renders_when_logged_in(self):
        self.client.login(username="tester", password="testpass123")
        r = self.client.get("/players/compare/")
        self.assertEqual(r.status_code, 200)

    def test_match_stars_renders_when_logged_in(self):
        self.client.login(username="tester", password="testpass123")
        r = self.client.get("/players/match-stars/?home=Arsenal&away=Chelsea")
        self.assertEqual(r.status_code, 200)

    def test_match_stars_rejects_same_team(self):
        self.client.login(username="tester", password="testpass123")
        r = self.client.get("/players/match-stars/?home=Arsenal&away=Arsenal")
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "différentes")

    def test_at_risk_renders_when_logged_in(self):
        self.client.login(username="tester", password="testpass123")
        r = self.client.get("/players/at-risk/?home=Arsenal&away=Chelsea")
        self.assertEqual(r.status_code, 200)